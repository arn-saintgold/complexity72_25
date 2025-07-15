#!/bin/bash

# Run topic modeling on cop26 tweets, and save the output to cop26.out, redirecting error to standard input
nohup python3.12 scripts/parallel_topic_model.py data/raw/cop26_tweets_en.parquet text > cop26.out 2>&1

# Run the same script on covid tweets and save it to covid.out
nohup python3.12 scripts/parallel_topic_model.py data/raw/covid_tweets_en.parquet text > covid.out 2>&1

# Run it on ukraine data and save output to ukraine.out
nohup python3.12  scripts/parallel_topic_model.py data/raw/ukraine_tweets_en.parquet text > ukraine.out 2>&1

