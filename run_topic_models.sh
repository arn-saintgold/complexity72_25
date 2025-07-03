#!/bin/bash

# Run the first Python script and save its output to out.out
python3.12 scripts/parallel_topic_model.py data/raw/covid_tweets_en.parquet text 

# Run the second Python script and save its output to tuo.out
python3.12  scripts/parallel_topic_model.py data/raw/ukraine_tweets_en.parquet text 

