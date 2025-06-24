# This script sets up a directory structure for a data science project.
# It creates the following directories:
# - data/raw: for raw data files
# - data/processed: for processed data files
# - notebooks: for Jupyter notebooks
# - scripts: for Python scripts
# - models: for machine learning models
# - reports: for reports and documentation
# - plots: for visualizations and plots

import os

def setup_workspace(directory_path):
    os.makedirs(os.path.join(directory_path,"data"), exist_ok=True)
    os.makedirs(os.path.join(directory_path,"data", "raw"), exist_ok=True)
    os.makedirs(os.path.join(directory_path,"data", "processed"), exist_ok=True)
    os.makedirs(os.path.join(directory_path,"notebooks"), exist_ok=True)
    os.makedirs(os.path.join(directory_path,"scripts"), exist_ok=True)
    os.makedirs(os.path.join(directory_path,"models"), exist_ok=True)
    os.makedirs(os.path.join(directory_path,"reports"), exist_ok=True)
    os.makedirs(os.path.join(directory_path,"plots"), exist_ok=True)
    print(f"Workspace directory '{directory_path}' has been set up.")

def main():
    directory_path = os.getcwd()  # Get the current working directory
    setup_workspace(directory_path)

if __name__ == "__main__":
    main()

