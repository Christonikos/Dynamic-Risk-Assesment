"""
    Script to deploy the latest model and reports to the deployment directory.
    This script handles argument parsing, logging configuration, and wandb run configuration.
"""
import logging
import os
import shutil
import argparse
import wandb
import time
import utils


####################function for deployment
def main(args):
    """
    Main function that deploys the latest model and reports to the deployment directory.

    Args:
        args: An object containing the necessary arguments and configurations.
    """
    # Log the start of the deployment process
    args.logging.info("Starting model deployment...")
    start_time = time.time()

    def find_latest_file(path):
        """
        Find the latest file in a given directory based on creation time.

        Args:
            path (str): The path to the directory.

        Returns:
            str: The full path of the latest file.
        """
        files = os.listdir(path)
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(path, x)))
        return os.path.join(path, latest_file)

    # Initialize a dictionary to hold the directories
    dir_collector = {}

    # Get the fname of the latest model
    path2models = os.path.join("..", args.config["output_model_path"])
    dir_collector["model"] = find_latest_file(path2models)

    # Get the fname of the latest .txt files (reports)
    path2reports = os.path.join("..", "reports")
    for report_type in ["scoring", "data_ingestion"]:
        report_dir = os.path.join(path2reports, report_type)
        dir_collector[f"{report_type}_report"] = find_latest_file(report_dir)

    # copy to the deployment directory
    depl_dir = os.path.join("..", args.config["prod_deployment_path"])
    if not os.path.exists(depl_dir):
        os.makedirs(depl_dir)

    for _, v in dir_collector.items():
        shutil.copy(v, depl_dir)

    # Log the deployment directory and dir_collector
    args.logging.info(f"Model and reports deployed to {depl_dir}")
    wandb.log({"deployment_directory": depl_dir})
    wandb.log({"deployed_files": dir_collector})
    # Log the time taken for the deployment process
    time_taken = time.time() - start_time
    wandb.log({"deployment_time_seconds": time_taken})

    # Log the latest model and report files as artifacts
    artifact = wandb.Artifact("deployed_files", type="deployment")
    for file_type, file_path in dir_collector.items():
        artifact.add_file(file_path, name=file_type)
    args.run.log_artifact(artifact)

    # Log metadata of the deployed model and reports
    metadata = {
        f"{file_type}_size_bytes": os.path.getsize(file_path),
        f"{file_type}_ctime": os.path.getctime(file_path),
    }
    for file_type, file_path in dir_collector.items():
        wandb.log(metadata)
    args.run.finish()


# %%
if __name__ == "__main__":
    # Define the ArgumentParser
    parser = argparse.ArgumentParser(description="Script to deploy the model")

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
        job_type="model_deployment",
        group="model_deployment",
        name=f"run_{utils.get_datetime()}_model_deployment",
    )
    # cast the params to the args object
    args.run = run
    args.config = config
    args.logging = logging
    # call the main
    main(args)
