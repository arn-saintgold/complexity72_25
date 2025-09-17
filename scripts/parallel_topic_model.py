# Script that takes a filename, a column name, and optional column names as parameters,
#  and performs topic modeling on the text data in the file.
# The optional columns are used for temporal or categorical analysis.
# It uses fast_hdbscan for a fully parallelized hdbscan implementation
#  and hdbscan's validity index function for parameter selection.
#
# Validity Index references:
# Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and Sander, J.,
# 2014. Density-Based Clustering Validation. In SDM (pp. 839-847).
#
# fast_hdbscan github:
# https://github.com/TutteInstitute/fast_hdbscan

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))  # Adjust as needed
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "scripts")))
import my_text_cleaning as tc

import logging
import random
from bertopic import __version__ as bertopic_version
from sys import version as python_version
import time
import argparse
from typing import Tuple, List
import pandas as pd
import numpy as np
from umap import UMAP

# from sklearn.decomposition import PCA
from bertopic.dimensionality import BaseDimensionalityReduction as umap_was_precomputed
from fast_hdbscan import HDBSCAN
#from hdbscan import validity_index
from scripts.my_validity_index import validity_index # Custom version to avoid numba issues
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.DEBUG,  # or DEBUG, WARNING, etc.
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("bertopic").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

DEBUGGING = False  # if True, skips parameter search and performs multithreaded umap
DEVICE = "cpu"

# Try different random seeds
# Needed to take into account UMAP multithreading stochastic behaviour and race conditions.
# Computing UMAP with a random seed may take a lot of time
RANDOM_SEEDS = [random.randint(0, 2**32 - 1) for _ in range(5)]


def search_params(embeddings: np.ndarray) -> List:
    global DEBUGGING
    global RANDOM_SEEDS
    logger.info("STARTING PARAMETER SELECTION")
    max_validity_value = -float("Inf")
    best_params = None

    for n_neighbors in [15]:
        for n_components in [
            50
        ]:  # 50 is the highest suggested number of components HDBSCAN can handle while taking a reasonable ammount of time to compute.
            logger.info(f"{n_components=}, {n_neighbors=}")
            t0 = time.time()

            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
                n_jobs=-1,
            )
            # No random seed search on parameter search, just average the values. Search later for random seed.

            many_reduced_embeddings = [
                umap_model.fit_transform(embeddings) for _ in RANDOM_SEEDS
            ]

            t1 = time.time()
            logger.info(f"EMBEDDING REDUCED IN {round(t1 - t0, 1)} SECONDS")
            for min_cluster_size in np.linspace(50, 500, 10).astype(int):
                for cluster_selection_method in ["eom"]:# ["eom", "leaf"]:
                    logger.info(
                        f"CLUSTERING WITH PARAMETERS: {min_cluster_size=}, {cluster_selection_method=}"
                    )
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        cluster_selection_method=cluster_selection_method,
                        metric="euclidean",
                    )
                    t0 = time.time()
                    many_labels = [
                        hdbscan_model.fit_predict(reduced_embeddings)
                        for reduced_embeddings in many_reduced_embeddings
                    ]

                    t1 = time.time()
                    logger.info(f"CLUSTERING FINISHED IN {round(t1 - t0, 1)} SECONDS")
                    logger.info("STARTING VALIDATION")

                    many_validity_values = []

                    many_reduced_embeddings = [arr.astype(np.float64, copy=False) for arr in many_reduced_embeddings]

                    many_validity_values = Parallel(n_jobs=-1)(
                        delayed(validity_index)(emb, labels)
                        for emb, labels in zip(many_reduced_embeddings, many_labels)
                    )

                    validity_value = np.mean(many_validity_values)
                    t2 = time.time()
                    logger.info(f"VALIDATION FINISHED IN {round(t2 - t1, 1)} SECONDS")
                    logger.info(f"BOTH FINISHED IN {round(t2 - t0, 1)} SECONDS")
                    logger.info(f"VALIDITY INDEX: {validity_value}")

                    if validity_value > max_validity_value:
                        best_RAND = None  # Will be assigned later
                        max_validity_value = validity_value
                        best_params = (
                            n_neighbors,
                            n_components,
                            min_cluster_size,
                            cluster_selection_method,
                            best_RAND,
                        )
    logger.info(f"SEARCHING RANDOM SEED IN {RANDOM_SEEDS}")
    # computing many umap reductions to find a good random seed for the parameters found during the search.
    many_umaps = [
        UMAP(
            n_neighbors=best_params[0],
            n_components=best_params[1],
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
        )
        for random_state in RANDOM_SEEDS
    ]
    hdbscan_model = HDBSCAN(
        min_cluster_size=best_params[2],
        cluster_selection_method=best_params[3],
        metric="euclidean",
    )

    many_reduced_embeddings = [
        this_umap.fit_transform(embeddings) for this_umap in many_umaps
    ]
    many_labels = [
        hdbscan_model.fit_predict(reduced_embeddings)
        for reduced_embeddings in many_reduced_embeddings
    ]

    many_reduced_embeddings = [arr.astype(np.float64, copy=False) for arr in many_reduced_embeddings]

    many_validity_values = Parallel(n_jobs=-1)(
        delayed(validity_index)(emb, labels)
        for emb, labels in zip(many_reduced_embeddings, many_labels)
    )

    many_validity_values = np.array(
        [
            validity_index(reduced_embeddings.astype(np.float64), labels)
            for (reduced_embeddings, labels) in zip(
                many_reduced_embeddings, many_labels
            )
        ]
    )
    # picking random seed with best performance
    best_RAND = RANDOM_SEEDS[
        many_validity_values.argmax()
    ]  # Random seed of the embedding with the best value among those with highest average validity value
    best_params = list(best_params)
    best_params[-1] = best_RAND
    logger.info(
        f"Best parameters: n_neighbors={best_params[0]}, n_components={best_params[1]}, min_cluster_size={best_params[2]}, cluster_selection_method={best_params[3]}, best_avg_validity_value={max_validity_value}, best seed={best_RAND}"
    )
    return best_params


def deduplicate_text_and_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray | None,
    col_name: str,
    keep: str | None = "first",
) -> Tuple[pd.DataFrame, np.ndarray | None]:
    """
    Returns a dataframe and corresponding embedding without duplicates and empty strings.
    Args:
        df (pandas.DataFrame): Dataframe with text to extract
        embeddings (numpy.ndarray): Embeddings of the texts to extract
        col_name (str): Name of the text column in df
        keep (str): What to do with the duplicates: 'first' to keep the first instance,
                    'last' to keep the last, None to drop every duplicate
    Returns:
        tuple(pd.DataFrame, np.ndarray | None): A tuple of the new text and corresponding embeddings.
    """
    # remove rows with empty strings in the text column
    non_empty_mask = df[col_name].astype(str).str.strip() != ""

    # deduplicate (after removing empty)
    unique_mask = ~df[non_empty_mask].duplicated(col_name, keep=keep)

    # combine masks
    final_mask = non_empty_mask & unique_mask.reindex(df.index, fill_value=False)

    unique_rows = df[final_mask]

    if embeddings is not None:
        unique_embeddings = embeddings[final_mask]
        return unique_rows, unique_embeddings
    else:
        return unique_rows, None


def topic_modeling(
    filename: str,
    text_column: str,
    embedding_model_name="all-MiniLM-L6-v2",
    *args,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO If UMAP parameters are fixed, compute embeddints right away.

    global DEBUGGING
    global DEVICE

    # Load Precomputed embeddings and transformer model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    actual_filename = filename.split("/")[-1]
    embedding_path = os.path.join("data", "processed", actual_filename + ".npy")
    logger.info(f"getting embeddings from {embedding_path}")
    embeddings = (
        np.load(embedding_path)  # , allow_pickle=True).item()
        if os.path.exists(embedding_path)
        else None
    )
    if embeddings is None:
        raise ValueError("EMBEDDINGS NOT FOUND")  # logger.info('ENCODING TEXT')
        # embeddings = embedding_model.encode(texts)
    else:
        logger.info("PRECOMPUTED EMBEDDINGS FOUND")

    # Load the data
    if filename.endswith(".parquet"):
        df = pd.read_parquet(filename)
    elif filename.endswith("csv"):
        df = pd.read_csv(filename)
    else:
        raise ValueError("Extension not recognized")
    # ? Take subset of data
    # ? Removed for actual analysis

    # Check if the specified text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in the DataFrame.")

    # Choose unique texts and embeddings
    clean_df = tc.clean_dataframe(
        df,
        text_column,
        phrases_to_remove=["&gt;", "&lt;", "&amp;", "RT : "],
        remove_empty=False,
        remove_urls=True,
        normalize_hashtags=True,
        normalize_mentions=True,
        user_placeholder="user",
        strip_punctuation=False,
        lowercase=False,
    )
    unique_df, unique_embeddings = deduplicate_text_and_embeddings(
        clean_df[["Clean" + text_column, text_column]],
        embeddings,
        "Clean" + text_column,
    )
    unique_texts, original_texts = (
        unique_df["Clean" + text_column],
        unique_df[text_column],
    )
    assert len(original_texts) == len(unique_texts) == len(unique_embeddings), (
        f"A problem occurred while cleaning texts: {len(original_texts)=}, {len(unique_texts)=}, {len(unique_embeddings)=}"
    )
    # Free some spce
    embeddings = None
    df = None

    # Search for the best parameters for UMAP and HDBSCAN
    if not DEBUGGING:
        best_params = search_params(unique_embeddings)

    else:
        best_params = [50, 5, 200, "eom", 0]  # search_params(unique_embeddings)

    umap_model = UMAP(
        n_neighbors=best_params[0],
        n_components=best_params[1],
        min_dist=0.0,
        metric="cosine",
        # n_jobs = -1,
        random_state=best_params[-1],
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=best_params[2],
        cluster_selection_method=best_params[3],
        metric="euclidean",
    )

    # ensure code is tested FAST
    if DEBUGGING:
        umap_model.random_state = None
        umap_model.n_jobs = -1

    # Create a BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_was_precomputed(),
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
        ctfidf_model=ClassTfidfTransformer(),
        # representation_model=unique_embeddings
        verbose=True,
        calculate_probabilities=True,
        language="english",
    )
    logger.info("REDUCING EMBEDDINGS")
    logger.debug(f"{unique_embeddings.shape = }")
    reduced_embeddings = umap_model.fit_transform(unique_embeddings)
    logger.info("EMBEDDING REDUCED")
    # Fit the model to the texts
    logger.info("CREATING TOPIC MODEL...")
    topics, _ = topic_model.fit_transform(unique_texts, embeddings=reduced_embeddings)

    logger.info("EVALUATING TOPIC MODEL...")
    try:
        validity_value = validity_index(reduced_embeddings.astype(np.float64), topics)
    except AttributeError as e:
        logging.error(e)
        logging.error("Converting topics to np.ndarray")
        validity_value = validity_index(
            reduced_embeddings.astype(np.float64), np.array(topics)
        )

    topic_info = topic_model.get_topic_info()
    document_info = topic_model.get_document_info(unique_texts)
    document_info["OriginalDocument"] = original_texts
    noise_percentage = len(document_info.query("Topic == -1")) / len(document_info)
    n_topics = len(topic_info) - 1
    model_info = f"{python_version = }\n{bertopic_version = }\nVALIDITY INDEX: {validity_value}\nUMAP Seed: {best_params[-1]}\nUMAP parameters: n_neighbours = {best_params[0]}, n_components = {best_params[1]}\nHDBSCAN parameters: min cluster size = {best_params[2]}, cluster selection method = {best_params[3]}\nN TOPICS: {n_topics}\nNOISE PERCENTAGE: {noise_percentage}."
    logging.info("MODEL INFO:\n" + model_info)
    model_info_path = os.path.join(".", "models", actual_filename + "_model_info.txt")
    logging.debug(f"MODEL INFO PATH: {model_info_path}")
    with open(model_info_path, "w") as handle:
        handle.write(model_info)

    logger.info(f"VALIDITY INDEX: {validity_value}")

    # Save the model
    logger.info("SAVING TOPIC MODEL...")
    topic_model.umap_model = (
        umap_model  # Assign actual UMAP model instead of the emtpy one.
    )
    model_filename = filename.split("/")[-1] + ".topic_model"
    model_path = os.path.join("models", model_filename.split()[-1])
    logger.debug(f"MODEL PATH: {model_path}")
    topic_model.save(model_path, serialization="safetensors")
    logger.info(f"Topic model saved as {model_filename}.")

    # Return dataset with topics
    return topic_model.get_topic_info(), topic_model.get_document_info(unique_texts)


# ? Was needed for dynamic topic modeling
# ? Not needed anymore
# ? def count_months_passed(df, col_name):
# ?     df[col_name] = pd.to_datetime(df[col_name])
# ?
# ?     min_date = df[col_name].min()
# ?     max_date = df[col_name].max()
# ?
# ?     # Calculate months difference
# ?     months_passed = (max_date.year - min_date.year) * 12 + (
# ?         max_date.month - min_date.month
# ?     )
# ?     return months_passed


def main():
    logger.info(f"{RANDOM_SEEDS = }")

    parser = argparse.ArgumentParser(description="Perform topic modeling on text data.")
    parser.add_argument(
        "filename", type=str, help="Path to the CSV file containing the text data."
    )
    parser.add_argument(
        "text_column", type=str, help="Name of the column containing the text data."
    )
    parser.add_argument(
        "--optional_columns",
        nargs="*",
        help="Optional columns for temporal or categorical analysis.",
    )

    args = parser.parse_args()

    logger.info(f'READING FILE "{args.filename}"')
    logger.info(f'TEXT COLUMN NAME "{args.text_column}"')

    topic_info, document_info = topic_modeling(
        args.filename, args.text_column, args.optional_columns
    )
    logger.info(f"N TOPICS: {len(topic_info) - 1}.")
    logger.info(
        f"Noise percentage: {round(100 * (len(document_info.query('Topic == -1')) / len(document_info)), 2)}%"
    )

    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    topic_info.to_csv(
        os.path.join(
            "data",
            "processed",
            "topic_info_" + args.filename.split("/")[-1].split(".")[0],
        )
        + ".csv"
    )
    document_info.to_csv(
        os.path.join(
            "data",
            "processed",
            "document_info_" + args.filename.split("/")[-1].split(".")[0],
        )
        + ".csv"
    )


if __name__ == "__main__":
    main()
