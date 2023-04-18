"""
This script creates a Flask web application that provides endpoints for
model predictions, scoring, summary statistics, and diagnostics.
"""

from flask import Flask, jsonify
import utils

from diagnostics_05 import (
    load_data,
    load_latest_production_model,
    model_predictions,
    execution_time,
    dataframe_summary,
    outdated_packages_list,
)

app = Flask(__name__)

# Load the model and test data using the config file
config = utils.load_config()
model = load_latest_production_model(config)
test_data = load_data(config)

# Prediction Endpoint
@app.route("/prediction", methods=["POST"])
def predict():
    """
    The prediction endpoint takes a dataset's file location as its input and returns the
    outputs of the prediction function.
    """
    predictions = model_predictions(model, test_data)
    return jsonify(predictions=predictions)


# Scoring Endpoint
@app.route("/scoring", methods=["GET"])
def score():
    """
    The scoring endpoint runs the scoring script and returns its output.
    """
    score = execution_time("scoring.py")
    return jsonify(score=score)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET"])
def stats():
    """
    The summary statistics endpoint runs the summary statistics function and returns its outputs.
    """
    summary = dataframe_summary(test_data)
    return jsonify(summary=summary)


# Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET"])
def diagnostics():
    """
    The diagnostics endpoint checks timing and percent NA values.
    """
    training_time = execution_time("training.py")
    ingestion_time = execution_time("ingestion.py")
    outdated_packages = outdated_packages_list()

    return jsonify(
        training_time=training_time,
        ingestion_time=ingestion_time,
        outdated_packages=outdated_packages,
    )


# Run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
