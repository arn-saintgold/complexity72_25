import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.getcwd(),
            "..",
        )
    )
)  # Adjust as needed
import argparse
import logging
from typing import Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import my_text_cleaning as tc
import torch

logging.basicConfig(
    level=logging.DEBUG,  # or DEBUG, WARNING, etc.
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
bertopic_logger = logging.getLogger("bertopic")
bertopic_logger.setLevel(logging.WARNING)
sentence_transformers_logger = logging.getLogger("sentence_transformers")
sentence_transformers_logger.setLevel(logging.WARNING)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

DEBUGGING = False  # if True, skips parameter search and performs multithreaded umap


DEVICE = "cuda"


def embed_cleaned_texts(
    filename: str,
    text_column: str,
    embedding_model_name="all-mpnet-base-v2",
    *args,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global DEBUGGING
    global DEVICE
    logger.info(f"GETTING EMBEDDER {embedding_model_name}")
    # Load Precomputed embeddings and transformer model
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA NOT AVAILABLE")
    embedding_model = SentenceTransformer(
        model_name_or_path=embedding_model_name, device=DEVICE
    )
    logger.debug(f"MODEL: {embedding_model}")
    if embedding_model._first_module() is None:
        logger.error('MODEL WAS NOT LOADED. LOADING "all-mpnet-base-v2"')
        embedding_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
        logger.debug(f"MODEL: {embedding_model}")
    assert embedding_model._first_module() is not None, "Model did not load properly"

    actual_filename = filename.split("/")[-1]
    embedding_path = os.path.join("data", "processed", actual_filename + ".npy")

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
    logger.info(f"DATAFRAME SIZE: {len(df)}")
    logger.info("CLEANING TEXTS")
    texts = tc.clean_dataframe(
        df,
        text_column,
        phrases_to_remove=["&gt;", "&lt;", "&amp;", "RT : "],
        remove_empty=False,
        normalize_hashtags=True,
        normalize_mentions=True,
        user_placeholder="user",
    )
    texts = [str(t) for t in texts[text_column].tolist()]
    logger.info(f"DATAFRAME SIZE AFTER CLEANING: {len(texts)}")
    logger.info("STARTING EMBEDDING")
    for batch_size in [2**i for i in range(10, -1, -1)]:
        try:
            logger.info(f"TRYING BATCH SIZE {batch_size}")
            embeddings = embedding_model.encode(
                texts, batch_size=batch_size, show_progress_bar=True
            )
            logger.info(f"SUCCESS WITH BATCH SIZE: {batch_size}")
            break
        except RuntimeError as e:
            logger.error(f"FAILED WITH BATCH SIZE {batch_size}")
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                logger.error("CUDA OUT OF MEMORY")
            else:
                raise e
    # batch_size = 16
    # embeddings = embedding_model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    with open(embedding_path, "wb") as handle:
        np.save(handle, embeddings)

    logger.info("EMBEDDING FINISHED")
    logger.info(f"SAVING EMBEDDINGS TO {embedding_path}")


def main():
    parser = argparse.ArgumentParser(description="Embed Texts.")
    parser.add_argument(
        "filename", type=str, help="Path to the CSV file containing the text data."
    )
    parser.add_argument(
        "text_column", type=str, help="Name of the column containing the text data."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=str,
        const="all-mpnet-base-v2",
        default="all-mpnet-base-v2",
        help="Embedding model to use.",
    )

    args = parser.parse_args()

    logger.info(f'READING FILE "{args.filename}"')
    logger.info(f'TEXT COLUMN NAME "{args.text_column}"')
    logger.info(f"EMBEDDING MODEL: {args.model}")
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)

    embed_cleaned_texts(
        filename=args.filename,
        text_column=args.text_column,
        embedding_model_name=args.model,
    )


if __name__ == "__main__":
    main()
