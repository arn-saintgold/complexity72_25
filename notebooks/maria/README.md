# Tweet Cleaning and Classification Pipeline

This repository contains a tweet cleaning pipeline and classification tools for analyzing tweets using transformer models. The goal is to prepare social media text for NLP tasks and classify it across sentiment, hate speech, and offensiveness dimensions.

## Cleaning Features

- HTML entity unescaping
- URL and mention removal
- Emoji normalization (using `emoji` library)
- Smart quote & non-ASCII cleanup
- Filters:
  - Very short tweets
  - Tweets with no alphabetic content
  - Hashtag-only or emoji-only tweets

## Classification Tasks (RoBERTa-based)

| Task               | Model Name                                               |
|--------------------|----------------------------------------------------------|
| **Sentiment**      | `cardiffnlp/twitter-roberta-base-sentiment-latest`       |
| **Hate Speech**    | `cardiffnlp/twitter-roberta-base-hate-multiclass-latest` |
| **Offensive**      | `cardiffnlp/twitter-roberta-base-offensive`              |

Each classifier:
- Returns predicted label 
- Returns softmax probabilities for each class
- Supports GPU acceleration with CUDA

## Usage

The pipeline is implemented in:

 [`data_cleaning_for_RoBERTa.ipynb`](notebooks/maria/data_cleaning_for_RoBERTa.ipynb)  
 [`inference_roberta_models.ipynb`](notebooks/maria/inference_roberta_models.ipynb)

Each notebook demonstrates how to clean tweets and run each classifier, with outputs stored in new DataFrame columns.

## Installation

```bash
pip install -r requirements.txt
