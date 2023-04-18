#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitity/Auxiliary scripts

@author: christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================
import datetime
import subprocess
import os
import pandas as pd
import logging
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from re import sub


# config loader
def load_config():
    """
    Load the configuration file of the project.
    """
    config_fname = os.path.join("..", "config.json")
    with open(config_fname, "r") as f:
        config = json.load(f)
    return config


def get_datetime():
    """
    Returns the datetime in a snake case format that can be used as a
    suffix for the logger name.
    """

    def snake_case(s):
        return "_".join(
            sub(
                "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))
            ).split()
        ).lower()

    # datetime object containing current date and time
    now = datetime.datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    date_time = snake_case(dt_string).replace(os.sep, "_").replace(":", "_")
    return date_time


def data_preprocessing(X):
    """
    Preprocess the input data by applying one-hot encoding to categorical
    values and standardization to numerical values.

    Parameters:
    - X: The input data (pandas DataFrame) containing corporation,
    lastmonth_activity, lastyear_activity, and
      number_of_employees columns.

    Returns:
    - preprocessed_df: A preprocessed pandas DataFrame with one-hot encoded
    categorical values and standardized
      numerical values.
    """

    df = pd.DataFrame(X)

    # One-hot-encoding for the 'corporation' column
    encoder = OneHotEncoder(sparse=False)
    corporation_one_hot = encoder.fit_transform(df[["corporation"]])
    corporation_labels = encoder.categories_[0]
    corporation_df = pd.DataFrame(corporation_one_hot, columns=corporation_labels)

    logging.info("One-hot encoding applied to 'corporation' column")

    # Standardization for the numerical columns
    numerical_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]
    scaler = StandardScaler()
    scaled_numerical_values = scaler.fit_transform(df[numerical_columns])
    scaled_numerical_df = pd.DataFrame(
        scaled_numerical_values, columns=numerical_columns
    )

    logging.info(
        "Standardization applied to numerical columns: %s", ", ".join(numerical_columns)
    )

    # Concatenate the one-hot-encoded and standardized dataframes
    preprocessed_df = pd.concat([corporation_df, scaled_numerical_df], axis=1)

    logging.info("Data preprocessing complete")

    return preprocessed_df


def run_app():
    subprocess.run(["python", "app.py"])


def run_apicalls():
    subprocess.run(["python", "apicalls.py"])
