import re
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from rampwf.workflows import ObjectDetector


class utils:
    """Utility functions helpful in the challenge."""

    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from ramp_custom.scoring import (
        ClassAveragePrecision,
        MeanAveragePrecision,
    )
    from ramp_custom.predictions import CustomPredictions
    from ramp_custom.images import load_image
    from ramp_custom import geometry
    from ramp_custom.geometry import apply_NMS_for_y_pred, apply_NMS_for_image


problem_title = "Follicle Detection and Classification"


# REQUIRED: Predictions, workflow, score_types, get_cv, get_train_data, get_test_data
Predictions = utils.CustomPredictions
workflow = ObjectDetector()
score_types = [
    utils.ClassAveragePrecision("Primordial"),
    utils.ClassAveragePrecision("Primary"),
    utils.ClassAveragePrecision("Secondary"),
    utils.ClassAveragePrecision("Tertiary"),
    utils.MeanAveragePrecision(
        class_names=["Primordial", "Primary", "Secondary", "Tertiary"]
    ),
]


def get_cv(X, y):
    """Split data by ovary number

    Uses LeaveOneGroupOut where each group is a set of images
    that correspond to a given overy number.

    Parameters:
    -----------
    X : np.array
        array of image absolute paths
    y : np.array
        array of lists of true follicule locations

    """

    def extract_ovary_number(filename):
        digit = re.match(r".*M0(\d)-\d.*", filename).group(1)
        return int(digit)

    groups = [extract_ovary_number(filename) for filename in X]
    cv = LeaveOneGroupOut()
    return cv.split(X, y, groups)


def _get_data(path=".", split="train"):
    """
    Returns
    X : np.array
        shape (N_images,)
    y : np.array
        shape (N_images,). Each element is a list of locations.

    """
    labels = pd.read_csv(os.path.join(path, "data", split, "labels.csv"))
    filepaths = []
    locations = []
    for filename, group in labels.groupby("filename"):
        filepath = os.path.join("data", split, filename)
        filepaths.append(filepath)

        locations_in_image = [
            {
                "bbox": (row["xmin"], row["ymin"], row["xmax"], row["ymax"]),
                "class": row["class"],
            }
            for _, row in group.iterrows()
        ]
        locations.append(locations_in_image)

    X = np.array(filepaths, dtype=object)
    y = np.array(locations, dtype=object)
    assert len(X) == len(y)
    if os.environ.get("RAMP_TEST_MODE", False):
        # launched with --quick-test option; only a small subset of the data
        X = X[[1, -1]]
        y = y[[1, -1]]
    return X, y


def get_train_data(path="."):
    """Get train data from ``data/train/labels.csv``

    Returns
    -------
    X : np.array
        array of shape (N_images,).
        each element in the array is an absolute path to an image
    y : np.array
        array of shape (N_images,).
        each element in the array if a list of variable length.
        each element in this list is a labelled location as a dictionnary::

            {"class": "Primary", "bbox": (2022, 8282, 2300, 9000)}

    """
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")
