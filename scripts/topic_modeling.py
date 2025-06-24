# Script that takes a filename, a column name, and optional column names as parameters,
# and performs topic modeling on the text data in the file.
# The optional columns are used for temporal or categorical analysis.

import os
import sys
import argparse
import pandas as pd
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


def search_params(embeddings):
    max_relative_validity = 0
    best_params = None
    for n_neighbors in [15, 50]:
        for n_components in [2, 5, 10, 20, 50]:
            umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric='cosine')
            reduced_embeddings = umap_model.fit_transform(embeddings)
            for min_cluster_size in [10, 20, 50, 100, 200, 500]:
                for cluster_selection_method in ['eom', 'leaf']:
                    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method=cluster_selection_method, prediction_data=True, gen_min_span_tree=True)
                    relative_validity = hdbscan_model.fit(reduced_embeddings).relative_validity_
                    if relative_validity > max_relative_validity:
                        max_relative_validity = relative_validity
                        best_params = (n_neighbors, n_components, min_cluster_size, cluster_selection_method,)
    print(f"Best parameters: n_neighbors={best_params[0]}, n_components={best_params[1]}, min_cluster_size={best_params[2]}, cluster_selection_method={best_params[3]}, max_relative_validity={max_relative_validity}")
    return(best_params)

def clean_dataframe(df, col_name):
    return df[df[col_name].str.len()>0]

def topic_modeling(filename, text_column, *args, **kwargs):
    # Load the data
    if filename.endswith('.parquet'):
        df = clean_dataframe(pd.read_parquet(filename), text_column)
    elif filename.endswith('csv'):
        df = clean_dataframe(pd.read_csv(filename),text_column)
    else:
        raise ValueError("Extension not recognized")
    # Take subset of data

    # embedding_model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    # Check if the specified text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in the DataFrame.")
    
    
    # Extract the text data
    texts = df[text_column].astype(str).tolist()
    
    embeddings = np.load(filename+'.npy', allow_pickle=True).item() if os.path.exists(filename+'.npy') else None
    if embeddings is None: 
        embeddings = embedding_model.encode(texts)
    print('SUCCESS')
    exit()

    # Create a CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    
    # Search for the best parameters for UMAP and HDBSCAN
    best_params = search_params(embeddings)
    umap_model = UMAP(n_neighbors=best_params[0], n_components=best_params[1], min_dist=0.0, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=best_params[2], metric='euclidean', cluster_selection_method=best_params[3], prediction_data=True, gen_min_span_tree=True) 
    # Create a UMAP model for dimensionality reduction
    umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine')
    
    # Create an HDBSCAN model for clustering
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
    
    # Create a BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model = hdbscan_model,
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words='english'),
        ctfidf_model=ClassTfidfTransformer(),
        #representation_model=

        verbose=True,
        calculate_probabilities=True,
        language='english',
    )

    topic_model = BERTopic(vectorizer_model=ClassTfidfTransformer(), umap_model=umap_model, hdbscan_model=hdbscan_model,
                           calculate_probabilities=True, verbose=True)
    # Fit the model to the texts
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    
    # Save the model
    model_filename = os.path.splitext(filename)[0] + '_topic_model'
    topic_model.save(model_filename)    
    print(f"Topic model saved to {model_filename}.")


    return topics, topic_model.get_topic_info()

def count_months_passed(df, col_name):


    df[col_name] = pd.to_datetime(df[col_name])

    min_date = df[col_name].min()
    max_date = df[col_name].max()

    # Calculate months difference
    months_passed = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
    return(months_passed)

def main():
    parser = argparse.ArgumentParser(description="Perform topic modeling on text data.")
    parser.add_argument("filename", type=str, help="Path to the CSV file containing the text data.")
    parser.add_argument("text_column", type=str, help="Name of the column containing the text data.")
    parser.add_argument("--optional_columns", nargs='*', help="Optional columns for temporal or categorical analysis.")

    
    args = parser.parse_args()

    print(f"Reading file: {args.filename}\nText column: {args.text_column}")

    topics, topic_info = topic_modeling(args.filename, args.text_column, args.optional_columns)
    print(f"Identified {len(set(topics))} topics.")
    print("Topics identified:")
    print(topic_info)

if __name__ == '__main__':
    main()