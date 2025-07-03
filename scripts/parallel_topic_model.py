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
import time
import argparse
import pandas as pd
import numpy as np
from umap import UMAP
#from sklearn.decomposition import PCA
from fast_hdbscan import HDBSCAN
from hdbscan import validity_index
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

DEBUG = False
DEVICE = 'cpu'

def search_params(embeddings):
    global DEBUG
    print("Starting Parameter selection")
    max_validity_value = -float("Inf")
    best_params = None

    for n_neighbors in [15]:#, 50]:
        for n_components in [5]:#, 20, 50]:
            print(f"{n_components=}, {n_neighbors=}")
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
                n_jobs=-1,                  # TODO Comment away later for reproducibility
                #random_state=1138341792,   # TODO Uncomment later for reproducibility
            )
            print("REDUCING EMBEDDINGS")
            t0 = time.time()
            reduced_embeddings = umap_model.fit_transform(embeddings)
            t1 = time.time()
            print(f"EMBEDDING REDUCED IN {round(t1-t0,1)} SECONDS")
            for min_cluster_size in [100, 150, 200, 500]:
                for cluster_selection_method in ["eom", "leaf"]:
                    print(f"CLUSTERING WITH PARAMETERS: {min_cluster_size=}, {cluster_selection_method=}")
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        cluster_selection_method=cluster_selection_method,
                        metric="euclidean",
                    )
                    t0 = time.time()
                    labels = hdbscan_model.fit_predict(reduced_embeddings)
                    t1 = time.time()
                    validity_value = validity_index(reduced_embeddings.astype(np.float64), labels)
                    t2 = time.time()
                    print(f"CLUSTERING FINISHED IN {round(t1-t0,1)} SECONDS")
                    print(f"VALIDATION FINISHED IN {round(t2-t1,1)} SECONDS")
                    print(f"BOTH FINISHED IN {round(t2-t0,1)} SECONDS")
                    print(f"VALIDITY INDEX: {validity_index}")

                    if validity_value > max_validity_value:
                        max_validity_value = validity_value
                        best_params = (
                            n_neighbors,
                            n_components,
                            min_cluster_size,
                            cluster_selection_method,
                        )
    print(
        f"Best parameters: n_neighbors={best_params[0]}, n_components={best_params[1]}, min_cluster_size={best_params[2]}, cluster_selection_method={best_params[3]}, max_validity_value={max_validity_value}"
    )
    return best_params


def clean_dataframe(df, embeddings, col_name):

    unique_mask = df[col_name].map(df[col_name].value_counts()) == 1
    unique_rows = df[unique_mask]
    unique_texts = unique_rows[col_name]
    
    # Select corresponding rows from the embeddings array
    unique_embeddings = embeddings[unique_mask.to_numpy()]
    return unique_texts, unique_embeddings


def topic_modeling(
    filename, text_column, embedding_model_name="all-MiniLM-L6-v2", *args, **kwargs
):
    # TODO If UMAP parameters are fixed, compute embeddints right away.

    global DEBUG
    global DEVICE

    # Load Precomputed embeddings and transformer model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    actual_filename = filename.split('/')[-1]
    embedding_path = os.path.join('data','processed',actual_filename+'.npy')
    print(f"getting embeddings from {embedding_path}")
    embeddings = (
        np.load(embedding_path)  # , allow_pickle=True).item()
        if os.path.exists(embedding_path)
        else None
    )
    if embeddings is None:
        raise ValueError("EMBEDDINGS NOT FOUND")  # print('ENCODING TEXT')
        embeddings = embedding_model.encode(texts)
    else:
        print("PRECOMPUTED EMBEDDINGS FOUND")
    
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
    
    # TODO Remove old code
    # Extract the text data
    # texts = df[text_column].astype(str).tolist()

    # Choose unique texts and embeddings
    unique_texts, unique_embeddings = clean_dataframe(df, embeddings, text_column)
    
    # Free some spce
    embeddings = None
    df = None

    # Search for the best parameters for UMAP and HDBSCAN
    if not DEBUG:
        best_params = search_params(unique_embeddings)
        
    else:
        best_params = [15,5,15,'eom']#search_params(unique_embeddings)
    umap_model = UMAP(
        n_neighbors=best_params[0],
        n_components=best_params[1],
        min_dist=0.0,
        metric="cosine",
        n_jobs = -1,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=best_params[2],
        cluster_selection_method=best_params[3],
        metric="euclidean",
    )

    # Create a BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
        ctfidf_model=ClassTfidfTransformer(),
        # representation_model=unique_embeddings
        verbose=True,
        calculate_probabilities=True,
        language="english",
    )

    # topic_model = BERTopic(vectorizer_model=ClassTfidfTransformer(), umap_model=umap_model, hdbscan_model=hdbscan_model,
    #                       calculate_probabilities=True, verbose=True)

    # Fit the model to the texts
    topics, _ = topic_model.fit_transform(unique_texts, embeddings=unique_embeddings)

    # TODO add final topics' validity index

    # Save the model
    model_filename = filename.split("/")[-1] + ".topic_model"
    model_path = os.path.join("models", model_filename.split()[-1])
    topic_model.save(model_path, serialization="safetensors")
    print(f"Topic model saved to {model_filename}.")

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

    print(f"Reading file: {args.filename}\nText column: {args.text_column}")

    topic_info, document_info = topic_modeling(
        args.filename, args.text_column, args.optional_columns
    )
    print(f"Identified {len(topic_info) - 1} topics.")
    print(
        f"Noise percentage: {round(100 * (len(document_info.query('Topic == -1')) / len(document_info)), 2)}%"
    )
    # print(topic_info)

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
