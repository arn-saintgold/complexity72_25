import pandas as pd
import os

df = pd.read_parquet(os.path.join('data','raw','ukraine_tweets_en.parquet'))

df = df.iloc[0:1000]

df.to_parquet(os.path.join('data','processed','toy_subset.parquet'))
