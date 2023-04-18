"""

This script contains functions for model and data diagnostics. The goal of the
 diagnostics is to find and understand
any problems that might exist in the model and data, allowing for a quick
resolution to ensure optimal model
performance.

"""

import pandas as pd
import timeit
import os
import argparse
import subprocess
import sys
import logging
import wandb
import pickle
from operator import itemgetter
from pathlib import Path
import utils


################## Function to load the production model


def load_latest_production_model(config):
    """
    Load the latest production model from the directory specified in the config.

    :param config: A dictionary containing the configuration.
    :return: The loaded model.
    """
    production_dir = os.path.join("..", config["prod_deployment_path"])

    # Find all model files in the production directory
    model_files = [f for f in os.listdir(production_dir) if f.endswith(".pkl")]

    if not model_files:
        raise FileNotFoundError("No model file found in the production directory.")

    # Get the last modification time for each model file
    model_files_mtime = [
        (f, os.path.getmtime(os.path.join(production_dir, f))) for f in model_files
    ]

    # Sort the model files by last modification time in descending order
    model_files_mtime.sort(key=itemgetter(1), reverse=True)

    # Select the latest model file
    latest_model_file = model_files_mtime[0][0]

    model_path = os.path.join(production_dir, latest_model_file)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


################## Function to load the test data
def load_data(config):
    """
    Load test data from a CSV file in the directory specified in the config.

    :param config: A dictionary containing the configuration.
    :return: A pandas DataFrame containing the test data.
    """
    test_data_path = Path("..", "data", config["test_data_path"])
    csv_files = list(test_data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the test data directory.")
    elif len(csv_files) > 1:
        raise RuntimeError(
            "Multiple CSV files found in the test data directory. Expected only one."
        )

    csv_path = csv_files[0]

    # Read the CSV file into a pandas DataFrame
    test_data = pd.read_csv(csv_path)

    return test_data


################## Function to get model predictions
def model_predictions(model, data):
    # preprocesses the data
    X = data
    y = data.pop("exited")

    X_prep = utils.data_preprocessing(X)

    # calculate the F1-score using the trained model
    model.fit(X_prep, y)
    predictions = model.predict(X_prep)
    return predictions.tolist()


################## Function to get summary statistics
def dataframe_summary(dataframe):
    # Calculate summary statistics here
    summary = dataframe.describe().to_dict()
    return summary


################## Function to get percentage of missing data


def get_missing_data_percentage(config):
    """
    Calculate the percentage of missing (NA) values in each column of the
    dataset stored in the directory specified
    in the config.

    :param config: A dictionary containing the configuration.
    :return: A list of percentages, with each element corresponding to the
    percentage of missing values in a
             particular column.
    """
    output_dir = Path("..", "data", config["test_data_path"])

    # Ensure only one CSV file is found
    csv_files = list(output_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in the output folder.")
    elif len(csv_files) > 1:
        raise RuntimeError(
            "Multiple CSV files found in the output folder. Expected only one."
        )

    csv_path = csv_files[0]

    # Read the CSV file into a pandas DataFrame
    dataset = pd.read_csv(csv_path)

    # Calculate the percentage of missing values in each column
    missing_data_percent = (dataset.isna().sum() / len(dataset)) * 100

    return missing_data_percent.tolist()


################## Function to get timings
def execution_time(script_name):
    # Calculate the execution time of a Python script
    start_time = timeit.default_timer()
    subprocess.run([sys.executable, script_name])
    end_time = timeit.default_timer()
    return end_time - start_time


################## Function to check dependencies
def outdated_packages_list():
    # Get a list of outdated packages
    outdated = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--outdated", "--format=freeze"]
    )
    outdated_packages = [pkg.decode("utf-8") for pkg in outdated.splitlines()]
    return outdated_packages


def main(args):
    """
    The main function of the script. Loads the production model and test data,
    calculates model predictions,
    data summary, execution times, and outdated packages. Logs the results and
    saves them as artifacts.

    :param args: An argparse object containing the arguments, logging,
    and wandb run objects.
    """

    # Load the model
    model = load_latest_production_model(args.config)
    args.logging.info("Loaded production model.")

    # Load the data
    test_data = load_data(args.config)
    args.logging.info("Loaded test data.")

    # Make predictions
    predictions = model_predictions(model, test_data)
    args.logging.info(f"Model predictions: {predictions}")

    # Get data summary
    summary = dataframe_summary(test_data)
    args.logging.info(f"Data summary: {summary}")

    # Get missing data %
    missing_data_percentage = get_missing_data_percentage(args.config)
    args.logging.info(f"Percentage of missing data: {missing_data_percentage}")

    # Get execution times
    training_time = execution_time("training.py")
    ingestion_time = execution_time("ingestion.py")
    args.logging.info(f"Execution Time (training.py): {training_time}")
    args.logging.info(f"Execution Time (ingestion.py): {ingestion_time}")

    # Get outdated packages
    outdated_packages = outdated_packages_list()
    args.logging.info(f"List of outdated packages: {outdated_packages}")

    # Log data to wandb
    args.run.log(
        {
            "training_time": training_time,
            "ingestion_time": ingestion_time,
            "summary": summary,
            "predictions": predictions,
        }
    )


if __name__ == "__main__":
    # Define the ArgumentParser
    parser = argparse.ArgumentParser(
        description="Script to run model and data diagnostics"
    )

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
        job_type="diagnostics",
        group="model_and_data_diagnostics",
        name=f"run_{utils.get_datetime()}_model_and_data_diagnostics",
    )
    # cast the params to the args object
    args.run = run
    args.config = config
    args.logging = logging
    # call the main
    main(args)
