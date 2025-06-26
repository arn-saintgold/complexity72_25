# Script that takes a filename, a column name, and optional column names as parameters,
# and performs topic modeling on the text data in the file.
# The optional columns are used for temporal or categorical analysis.

import os
import argparse
import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

DEBUG = False

def search_params(embeddings):
    global DEBUG
    print('Starting Parameter selection')
    if DEBUG:
        
        max_relative_validity = 0
        best_params = None
        for n_components in [5, 20, 50]:
            print(f'{n_components=}')
            umap_model = PCA(n_components)
            reduced_embeddings = umap_model.fit_transform(embeddings)
            for min_cluster_size in [15, 50, 100, 200]:
                for cluster_selection_method in ["eom", "leaf"]:
                    print(f'{min_cluster_size=}, {cluster_selection_method=}')
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        metric="euclidean",
                        cluster_selection_method=cluster_selection_method,
                        prediction_data=True,
                        gen_min_span_tree=True,
                    )
                    relative_validity = hdbscan_model.fit(
                        reduced_embeddings
                    ).relative_validity_
                    if relative_validity > max_relative_validity:
                        max_relative_validity = relative_validity
                        best_params = (
                            None,
                            n_components,
                            min_cluster_size,
                            cluster_selection_method,
                        )
        print(
            f"Best parameters: n_components={best_params[1]}, min_cluster_size={best_params[2]}, cluster_selection_method={best_params[3]}, max_relative_validity={max_relative_validity}"
        )
        return best_params
    max_relative_validity = 0
    best_params = None
    for n_neighbors in [15, 50]:
        for n_components in [5, 20, 50]:
            print(f'{n_components=}, {n_neighbors=}')
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
            )
            reduced_embeddings = umap_model.fit_transform(embeddings)
            for min_cluster_size in [15, 50, 100, 200]:
                for cluster_selection_method in ["eom", "leaf"]:
                    print(f'{min_cluster_size=}, {cluster_selection_method=}')
                    hdbscan_model = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        metric="euclidean",
                        cluster_selection_method=cluster_selection_method,
                        prediction_data=True,
                        gen_min_span_tree=True,
                    )
                    relative_validity = hdbscan_model.fit(
                        reduced_embeddings
                    ).relative_validity_
                    if relative_validity > max_relative_validity:
                        max_relative_validity = relative_validity
                        best_params = (
                            n_neighbors,
                            n_components,
                            min_cluster_size,
                            cluster_selection_method,
                        )
    print(
        f"Best parameters: n_neighbors={best_params[0]}, n_components={best_params[1]}, min_cluster_size={best_params[2]}, cluster_selection_method={best_params[3]}, max_relative_validity={max_relative_validity}"
    )
    return best_params


def clean_dataframe(df, col_name):
    return df[df[col_name].str.len() > 0]


def topic_modeling(
    filename, text_column, embedding_model_name="all-MiniLM-L6-v2", *args, **kwargs
):
    global DEBUG
    # Load the data
    if filename.endswith(".parquet"):
        df = clean_dataframe(pd.read_parquet(filename), text_column)
    elif filename.endswith("csv"):
        df = clean_dataframe(pd.read_csv(filename), text_column)
    else:
        raise ValueError("Extension not recognized")
    # Take subset of data

    # embedding_model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    # Check if the specified text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in the DataFrame.")

    # Extract the text data
    texts = df[text_column].astype(str).tolist()
    embedding_path = os.path.join('data','processed','covid_tweets_en.parquet.npy')
    print(f'getting embeddings from {embedding_path}')
    embeddings = (
        np.load(embedding_path)#, allow_pickle=True).item()
        if os.path.exists(embedding_path)
        else None
    )
    if embeddings is None:
        raise ValueError("EMBEDDINGS NOT FOUND") #print('ENCODING TEXT')
        embeddings = embedding_model.encode(texts)
    else:
        print('PRECOMPUTED EMBEDDINGS FOUND')

    # Create a CountVectorizer
    # vectorizer = CountVectorizer(stop_words="english")


    unique_rows = df[df[text_column].map(df[text_column].value_counts()) == 1]
    unique_mask = df[text_column].map(df[text_column].value_counts()) == 1

    # Select corresponding rows from the embeddings array
    unique_embeddings = embeddings[unique_mask.to_numpy()]
    unique_texts = unique_rows[text_column]

    # Search for the best parameters for UMAP and HDBSCAN
    if not DEBUG:
        best_params = search_params(embeddings)
        umap_model = UMAP(
            n_neighbors=best_params[0],
            n_components=best_params[1],
            min_dist=0.0,
            metric="cosine",
            low_memory=True,
        )
    else:
        best_params = search_params(embeddings)
        umap_model = PCA(best_params[1])
    hdbscan_model = HDBSCAN(
        min_cluster_size=best_params[2],
        metric="euclidean",
        cluster_selection_method=best_params[3],
        prediction_data=True,
        gen_min_span_tree=True,
    )
    # HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method=cluster_selection_method, prediction_data=True, gen_min_span_tree=True)

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

    # Save the model
    model_filename = filename.split("/")[-1] + ".topic_model"
    model_path = os.path.join("models", model_filename.split()[-1])
    topic_model.save(model_path, serialization='safetensors')
    print(f"Topic model saved to {model_filename}.")

    # TODO return dataset with topics
    return topic_model.get_topic_info(), topic_model.get_document_info(unique_texts)

def count_months_passed(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name])

    min_date = df[col_name].min()
    max_date = df[col_name].max()

    # Calculate months difference
    months_passed = (max_date.year - min_date.year) * 12 + (
        max_date.month - min_date.month
    )
    return months_passed


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
