#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
import wandb
import logging
import shutil
import argparse
import utils


def aggregate_feature_importances(feature_importances, columns, prefix):
    """
    Aggregate the feature importance scores of the one-hot encoded columns
    related to the original categorical column.

    Parameters:
    - feature_importances: A NumPy array containing the feature importance scores.
    - columns: A pandas Index containing the column names of the preprocessed data.
    - prefix: The prefix for the one-hot encoded columns related to the original
    categorical column.

    Returns:
    - aggregated_importance: The aggregated feature importance score of the
      original categorical column.
    """
    aggregated_importance = 0
    related_columns = [col for col in columns if col.startswith(prefix)]

    for col in related_columns:
        aggregated_importance += feature_importances[columns.get_loc(col)]

    return aggregated_importance


def get_most_important_features(estimators, X, n_features=5, prefix="corporation"):
    """
    Calculate the most important features by averaging the coefficients of the
    features across all folds and
    aggregating the importance scores of one-hot encoded columns related to the
    original categorical column.

    Parameters:
    - estimators: A list of fitted LogisticRegression estimators from each fold
    in cross-validation.
    - X: The preprocessed data (pandas DataFrame) used for training the model.
    - n_features: The number of top features to return (default: 5).
    - prefix: The prefix for the one-hot encoded columns related to the original
    categorical column (default: "corporation").

    Returns:
    - important_features: A pandas Series containing the aggregated importance
    scores of the original categorical
      column and the importance scores of other features, sorted in descending
      order of importance.
    """

    feature_importances = []

    for estimator in estimators:
        feature_importance = np.abs(estimator.coef_[0])
        feature_importances.append(feature_importance)

    # Calculate the average feature importances across all folds
    avg_feature_importance = np.mean(feature_importances, axis=0)

    # Get the column names of the preprocessed X
    preprocessed_columns = X.columns

    # Aggregate the feature importances of one-hot encoded columns related to
    # the original categorical column
    aggregated_importance = aggregate_feature_importances(
        avg_feature_importance, preprocessed_columns, prefix
    )

    # Create a pandas Series with feature importances
    feature_importance_series = pd.Series(
        avg_feature_importance, index=preprocessed_columns
    )

    # Remove one-hot encoded columns related to the original categorical column
    feature_importance_series = feature_importance_series.drop(
        labels=[col for col in preprocessed_columns if col.startswith(prefix)]
    )

    # Add the aggregated importance score of the original categorical column
    feature_importance_series[prefix] = aggregated_importance

    # Sort the feature importances in descending order
    important_features = feature_importance_series.sort_values(ascending=False)

    # Return the top n_features
    return important_features.head(n_features)


def train_model(args):
    """
    Trains a logistic regression model using the provided training data and
    logs the model and some relevant metrics to the W&B platform.

    Args:
        args: An object containing the following arguments:
            - training_data: A pandas dataframe containing the training data
                             (must include an "exited" column)
            - n_splits: The number of folds to use for cross-validation

    Returns:
        None
    """

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=0)

    y = args.training_data.pop("exited")
    X = args.training_data

    # preprocess data
    X_prep = utils.data_preprocessing(X)
    # train model
    cv_results = cross_validate(
        model, X_prep, y, cv=kf, scoring="roc_auc", return_estimator=True
    )

    # Get the 5 most important features
    important_features = get_most_important_features(
        cv_results["estimator"], X_prep, n_features=5, prefix="corporation"
    )

    fig, ax = plt.subplots()
    important_features.plot(kind="bar", title="Most important features", ax=ax)
    plt.savefig("important_features.png")

    # and store artifact to WANDB
    artifact = wandb.Artifact(
        name="important_features",
        type="figure",
        description="A bar plot of the 5 most important features.",
    )
    artifact.add_file("important_features.png")
    run.log_artifact(artifact)

    mean_auc = cv_results["test_score"].mean()
    std_auc = round(cv_results["test_score"].std(), 2)

    # log to W&B
    run.log({"mean_auc": mean_auc, "std_auc": std_auc})
    # log the results and add them to W&B
    args.logging.info(f"Mean AUC of model: {mean_auc} +/- {std_auc}")
    # set the path
    path_to_model_output = os.path.join("..", args.config["output_model_path"])
    if not os.path.exists(path_to_model_output):
        os.makedirs(path_to_model_output)
    fname = os.path.join(
        path_to_model_output, f"{args.run.project}_{utils.get_datetime()}.pkl"
    )

    model.fit(X_prep, y)
    with open(fname, "wb") as f:
        # store model locally
        pickle.dump(model, f)

        logging.info(f"Saved trained model @{path_to_model_output}")
    # and store artifact to WANDB
    artifact = wandb.Artifact(
        name="trained_model",
        type="sklearn model",
        description="The model trained with the training data.",
    )
    artifact.add_file(fname)
    run.log_artifact(artifact)
    logging.info(f"Logged trained model on WANDB in {run.name}")


def main(args):
    # fetch the training dataset
    artifact = args.run.use_artifact(
        os.path.join(
            config["project_name"], "deduped_dataframe:v0"
        )  # use the first version for training
    )
    # read the training data
    training_data = pd.read_csv(artifact.file())
    # cast to the args object
    args.training_data = training_data
    del training_data
    args.logging.info(
        f"""
                      Loaded artifact: {artifact.name} to train the ML model.
                      The description of the artifact is: {artifact.description}
                      """
    )
    # cleanup the artifacts directory
    shutil.rmtree("artifacts")
    # train the model and store artifacts to W&B
    train_model(args)


# %%
if __name__ == "__main__":
    # Define the ArgumentParser
    parser = argparse.ArgumentParser(description="Script to train the model")
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
        job_type="model_training",
        group="model_training",
        name=f"run_{utils.get_datetime()}_model_training",
    )
    # cast the params to the args object
    args.run = run
    args.config = config
    args.logging = logging
    # call the main
    main(args)

    run.finish()
