#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""
import pandas as pd
import logging
import os
from pathlib import Path
import wandb
import utils


# Data injection
def merge_multiple_dataframe(config, run):
    """
    Reads the separate .csvs available at the input directory,
    concatenates, and stores them as a single file in the output
    directory. Both directories defined at the configuration file.

    Parameters
    ----------
    config : json
        The basic configuration file of the project.

    run :  wandb.sdk.wandb_run.Run
        The initialized WANDB run object.

    Returns
    -------
    Concatenated pandas dataframe stored at the output direcory.

    """

    # read and concatenate data
    input_folder = os.path.join("..", "data", config["input_folder_path"])
    output_folder = Path(os.path.join("..", "data", config["output_folder_path"]))

    logging.info(f"Reading input data from {input_folder}")

    if not os.path.exists(input_folder):
        raise ValueError(f"Input directory {input_folder} does not exist")

    input_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    data_frames = [
        pd.read_csv(os.path.join(input_folder, file)) for file in input_files
    ]
    output_file = pd.concat(data_frames, axis=0)

    logging.info(f"Read {len(data_frames)} input files")
    logging.info(f"Concatenated {len(output_file)} rows into output file")

    # ensure no duplicated values exist
    deduped_df = output_file.drop_duplicates()

    n_duplicates = len(output_file) - len(deduped_df)
    logging.info(f"Removed {n_duplicates} duplicated rows")

    # store output data
    logging.info(f"Writing output file to {output_folder}")
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    output_file_path = output_folder / f"{utils.get_datetime()}_finaldata.csv"
    deduped_df.to_csv(output_file_path, index=False)

    logging.info("Finished writing output file")

    # and store artifact to WANDB
    artifact = wandb.Artifact(
        name="deduped_dataframe",
        type="pandas.core.frame.DataFrame",
        description="The merged dataframe after duplicated values have been removed.",
    )
    artifact.add_file(output_file_path)
    run.log_artifact(artifact)
    logging.info(f"Logged {output_file_path} dataframe on WANDB in {run.name}")

    reporting_dir = Path(os.path.join("..", "reports", "data_ingestion"))
    if not reporting_dir.is_dir():
        reporting_dir.mkdir(parents=True)

    fname = os.path.join(reporting_dir, f"ingestedfiles_{utils.get_datetime()}.txt")

    with open(fname, "w") as file:
        file.write(f"Ingestion date: {utils.get_datetime()}\n")
        for idx, file_name in enumerate(input_files):
            file.write(f"{file_name} \n")
            file.write(f"Dataframe shape: {data_frames[idx].shape} \n")
        file.write(f"Number of duplicated rows removed: {n_duplicates} \n")


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
        job_type="data_injestion",
        group="data_injestion",
        name=f"run_{utils.get_datetime()}_data_injestion",
    )

    # merge and store the multiple files
    merge_multiple_dataframe(config, run)
    run.finish()
