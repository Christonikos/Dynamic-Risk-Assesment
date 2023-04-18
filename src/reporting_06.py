import pickle
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import wandb
import argparse
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)
from sklearn.metrics import f1_score
from reportlab.lib.styles import getSampleStyleSheet
import utils


################## Function to load the production model
def load_production_model(config):
    """
    Load the production model from the directory specified in the config.

    :param config: A dictionary containing the configuration.
    :return: The loaded model.
    """
    production_dir = os.path.join("..", config["prod_deployment_path"])

    # Get all model files in the directory
    model_files = [f for f in os.listdir(production_dir) if f.endswith(".pkl")]

    if not model_files:
        raise FileNotFoundError("No model file found in the production directory.")

    # Sort the model files by their modification time and get the most recent one
    model_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(production_dir, f)), reverse=True
    )
    latest_model_file = model_files[0]
    latest_model_path = os.path.join(production_dir, latest_model_file)

    with open(latest_model_path, "rb") as f:
        model = pickle.load(f)

    return model


################## Function to load the test data
def load_data(config):
    """
    Load test data from a CSV file in the directory specified in the config.

    :param config: A dictionary containing the configuration.
    :return: A pandas DataFrame containing the test data.
    """
    dataset_csv_path = Path("..", "data", config["output_folder_path"])
    csv_files = list(dataset_csv_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in the test data directory.")

    # Sort the CSV files by their modification time and get the most recent one
    csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    latest_csv_path = csv_files[0]

    # Read the CSV file into a pandas DataFrame
    test_data = pd.read_csv(latest_csv_path)

    return test_data


############## Function for plotting the confusion matrix


def plot_confusion_matrix(cm, output_path):
    """
    Plot and save the confusion matrix to the given output path.

    :param cm: A confusion matrix.
    :param output_path: A string representing the output path where the confusion matrix image will be saved.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


############## Function for reporting


def score_model(model, data, target_column, config):
    """
    Calculate the confusion matrix using the test data and the deployed model,
    and save the confusion matrix plot to the workspace.

    :param model: A scikit-learn model that is already fitted.
    :param data: A pandas DataFrame containing the test data.
    :param target_column: A string indicating the name of the target variable column in the data.
    :param config: A dictionary containing the configuration.
    """
    # Split data into features (X) and target (y)
    X = data.drop(target_column, axis=1)
    y_true = data[target_column]

    # Make predictions using the model
    y_pred = model.predict(X)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Save the confusion matrix plot to the workspace
    output_dir = os.path.join("..", "images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the model name from the model object
    model_name = type(model).__name__

    # Append the model name to the output file name
    output_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    print(output_path)
    plot_confusion_matrix(cm, output_path)


############## Function for creating a .pdf report


def create_pdf_report(confusion_matrix_path, summary_stats, f1_score):
    """
    Create a PDF report with the confusion matrix plot, summary statistics,
    and other diagnostics.

    :param confusion_matrix_path: Path to the confusion matrix plot image file.
    :param summary_stats: Summary statistics of the data.
    :param f1_score: Model's F1 score.
    :param output_file: Output file path for the PDF report.
    """

    # Set up the output directory
    output_dir = os.path.join("..", "reports", "pdf_reports")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "scoring_and_diagnostics_report.pdf")

    # Set up the PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Add title
    story.append(Paragraph("Model and Data Diagnostics Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Add confusion matrix plot
    story.append(Paragraph("Confusion Matrix:", styles["Heading2"]))
    im = Image(confusion_matrix_path, 5 * inch, 4 * inch)
    story.append(im)
    story.append(Spacer(1, 12))

    # Add summary statistics
    story.append(Paragraph("Summary Statistics:", styles["Heading2"]))
    data = [["Statistic", "Value"]] + list(summary_stats.items())
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 14),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    # Add F1 score
    story.append(Paragraph("F1 Score:", styles["Heading2"]))
    story.append(Paragraph(str(f1_score), styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Read the ingestion information from the txt file
    ingestion_info_path = os.path.join("..", "reports", "data_ingestion")
    # load the last trained model
    paths = sorted(Path(ingestion_info_path).iterdir(), key=os.path.getmtime)
    with open(paths[-1], "r") as file:
        ingestion_info = file.read()
    ingestion_info = ingestion_info.replace("\n", "<br/>")

    # Add Data Injestion info
    story.append(Paragraph("Data Injestion Info:", styles["Heading2"]))
    story.append(Paragraph(str(ingestion_info), styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Build the PDF
    doc.build(story)


def main(args):
    # load data
    data = load_data(args.config)
    # load the model
    model = load_production_model(args.config)

    # preprocess data
    data_prep = utils.data_preprocessing(data)
    # cast back the target column
    data_prep["exited"] = data.exited.values.tolist()
    # score the model and plot the confusion matrix
    score_model(model, data_prep, "exited", args.config)

    confusion_matrix_path = os.path.join("..", "images", "confusionmatrix.png")
    summary_stats = data.describe()
    X = data_prep
    y = data_prep.exited

    # calculate the F1-score using the trained model
    model.fit(X, y)
    predicted_labels = model.predict(X)
    f1_ = f1_score(y, predicted_labels)

    # create a .pdf report
    create_pdf_report(confusion_matrix_path, summary_stats, f1_)


# %%
if __name__ == "__main__":
    # Define the ArgumentParser
    parser = argparse.ArgumentParser(description="Script to generate model reports")

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
        job_type="reporting",
        group="model_reporting",
        name=f"run_{utils.get_datetime()}_model_reporting",
    )
    # cast the params to the args object
    args.run = run
    args.config = config
    args.logging = logging
    main(args)
