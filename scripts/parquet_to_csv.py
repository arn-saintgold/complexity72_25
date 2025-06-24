import os
import pandas as pd

print("Converting Parquet files to CSV...")
# Check current directory
print("os.getcwd():", os.getcwd(), flush=True)


for filename in os.listdir('./data/raw'):
    print("Processing file:", filename, flush=True)
    if filename.endswith('.parquet'):
        current_path = os.path.join('./data/raw', filename)
        print("Current path:", current_path, flush=True)
        df = pd.read_parquet(current_path)

        # save parquet file in data/raw to csv in data/processed
        os.makedirs('./data/processed', exist_ok=True)
        new_filename = filename.replace('.parquet', '.csv')
        df.to_csv(f'./data/processed/{new_filename}', index=False)
        