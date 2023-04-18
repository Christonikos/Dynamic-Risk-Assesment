import os
import logging
import wandb
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.metrics import f1_score
import subprocess
from ingestion_01 import merge_multiple_dataframe
import utils


def _get_latest_file_fname(path, file_type):

    _files = [
        f for f in os.listdir(path) if not f.startswith(".") and f.endswith(file_type)
    ]
    latest_fname = os.path.join(
        path, max(_files, key=lambda x: os.path.getctime(os.path.join(path, x)),),
    )
    return latest_fname


def main():
    ################## Define Paths
    path2reports = os.path.join("..", "reports")
    path2injestion_reports = os.path.join(path2reports, "data_ingestion")
    path2source_files = os.path.join("..", "data", "sourcedata")
    path2injested_files = os.path.join("..", "data", "ingesteddata")
    path2production = os.path.join("..", "production_deployment")
    path2scores = os.path.join("..", "reports", "scoring")
    ################## Check and read new data
    # first, read ingestedfiles.txt

    latest_report_fname = _get_latest_file_fname(path2injestion_reports, ".txt")
    # for debugging
    latest_report_fname = os.path.join(
        path2injestion_reports, "ingestedfiles_09_03_2023_09_30_49.txt"
    )
    with open(latest_report_fname, "r") as file:
        injestion_report_content = [line.strip() for line in file]
    injested_files = [f for f in injestion_report_content if "dataset" in f]

    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_data_files = [
        f
        for f in os.listdir(path2source_files)
        if "dataset" in f and not f.startswith(".")
    ]

    # Convert the lists to sets for easy comparison
    source_data_files_set = set(source_data_files)
    injested_files_set = set(injested_files)

    # Find the files that are in source_data_files but not in injested_files
    new_files = list(source_data_files_set.difference(injested_files_set))

    ################## Proceed only if new data
    if len(new_files) > 0:
        print("New files found in source data files:")
        print(new_files)

        # injest the new dataframes
        config["input_folder_path"] = "sourcedata"
        merge_multiple_dataframe(config, run)

        # load the latest ingested file
        latest_ingested_fname = _get_latest_file_fname(path2injested_files, ".csv")
        new_dedupped_file = pd.read_csv(latest_ingested_fname)

        # load the latest pdoduction model
        latest_production_model_fname = _get_latest_file_fname(path2production, ".pkl")

        latest_score_fname = _get_latest_file_fname(path2scores, ".txt")
        with open(latest_score_fname, "r") as file:
            latest_score_content = []
            for line in file:
                content = line.strip()
                latest_score_content.append(content)

        with open(latest_production_model_fname, "rb") as f:
            production_model = pickle.load(f)

        # calculate the scores on the new dataset
        X = new_dedupped_file
        y = new_dedupped_file.pop("exited")
        X_prep = utils.data_preprocessing(X)
        predicted_labels = production_model.predict(X_prep)
        f1_new = f1_score(y, predicted_labels)

        # now get the f1 score from the production environment
        f1_old = float(re.findall(r"\d+\.*\d*", latest_score_content[-1])[0])

        ##################Checking for model drift
        if np.abs(f1_new - f1_old) > 0.1:
            logging.info("Model drift detected")
            # TODO: Report why
            ##################Re-deployment
            logging.info(" Running deployment_04.py...")
            subprocess.run(["python", "deployment_04.py"])

            ##################Diagnostics and reporting

            logging.info(" Running diagnostics_05.py...")
            subprocess.run(["python", "diagnostics_05.py"])

            logging.info(" Running reporting_06.py...")
            subprocess.run(["python", "reporting_06.py"])

        else:
            logging.info("No Model drift detected")

    else:
        print("No new files found in source data files.")


# %%
if __name__ == "__main__":
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
        job_type="full_process",
        group="full_process",
        name=f"run_{utils.get_datetime()}_full_process",
    )

    main()
    run.finish()
