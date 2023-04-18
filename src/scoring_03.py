"""
Model Scoring Script

This script loads input data from a specified test-data path, preprocesses it, and evaluates a trained model
using the preprocessed data.

The script calculates the F1-score and saves the results in a report file.

Functions:
    - load_input_data(config: dict) -> pd.DataFrame
    - load_model(config: dict) -> pd.DataFrame
    - main(args: argparse.Namespace)

Example usage:
    python model_scoring.py --config=config.yaml
"""

import pandas as pd
import os
from sklearn.metrics import f1_score
import logging
import argparse
import wandb
from pathlib import Path
import utils


def load_input_data(config):
    """
    Load input data from a CSV file located in the test-data path specified in the configuration.

    Args:
        config (dict): Dictionary containing the configuration settings.

    Returns:
        input_df (pd.DataFrame): Pandas DataFrame containing the input data.
    """
    # read data from the test-data path
    input_dir = os.path.join("..", "data", config["test_data_path"])
    file = [f for f in os.listdir(input_dir) if not f.startswith(".") and ".csv" in f]
    fname = os.path.join(input_dir, file[0])
    input_df = pd.read_csv(fname)
    return input_df


def load_model(config):
    """
    Load the last trained model from the output model path specified in the configuration.

    Args:
        config (dict): Dictionary containing the configuration settings.

    Returns:
        trained_model (pd.DataFrame): Pandas DataFrame containing the trained model.
    """
    input_dir = os.path.join("..", config["output_model_path"])
    # load the last trained model
    paths = sorted(Path(input_dir).iterdir(), key=os.path.getmtime)
    trained_model = pd.read_pickle(paths[-1])
    return trained_model


def main(args):
    """
    Main function for model scoring. Loads input data, preprocesses it, loads the trained model,
    calculates the F1-score, and saves the results in a report file.

    Args:
        args (argparse.Namespace): Parsed command line arguments containing the configuration.
    """
    # load input data
    input_df = load_input_data(args.config)
    # preprocess input data
    input_df_prep = utils.data_preprocessing(input_df)
    # load trained model
    trained_model = load_model(args.config)

    X = input_df_prep
    y = input_df.exited

    # calculate the F1-score using the trained model
    trained_model.fit(X, y)
    predicted_labels = trained_model.predict(X)
    f1 = f1_score(y, predicted_labels)

    reporting_dir = Path(os.path.join("..", "reports", "scoring"))
    if not reporting_dir.is_dir():
        reporting_dir.mkdir(parents=True)

    fname = os.path.join(reporting_dir, f"latestscore_{utils.get_datetime()}.txt")

    with open(fname, "w") as file:
        file.write(f"Scoring date: {utils.get_datetime()}\n")
        file.write(f"Testing dataset shape: {X.shape}\n")
        file.write(f"Model used: {trained_model}\n")
        file.write(f"F1 on test data: {f1}\n")


# %%
if __name__ == "__main__":
    # Define the ArgumentParser
    parser = argparse.ArgumentParser(description="Script to run model scoring")
    parser.add_argument("--n_splits", default=10)

    # Parse the arguments
    args = parser.parse_args()

    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # get the project config
    config = utils.load_config()

    # configure the wanbd run
    run = wandb.init(
        project=config["project_name"],
        job_type="model_scoring",
        group="model_scoring",
        name=f"run_{utils.get_datetime()}_model_scoring",
    )
    # cast the params to the args object
    args.run = run
    args.config = config
    args.logging = logging
    # call the main
    main(args)

    run.finish()
