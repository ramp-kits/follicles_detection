import re
import sys
import os
import pandas as pd
import numpy as np

from rampwf.workflows import ObjectDetector
from rampwf.prediction_types.detection import (
    Predictions as DetectionPredictions,
)
from sklearn.model_selection import LeaveOneGroupOut


class utils:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from ramp_custom.scores import (
        ClassAveragePrecision,
        MeanAveragePrecision,
        apply_NMS_for_y_pred,
    )


problem_title = "Follicle Detection and Classification"


class CustomPredictions(DetectionPredictions):
    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Combine multiple predictions into a single one.

        This is used when the "bagged scores" are computed.

        Parameters
        ----------
        predictions_list : list
            list of CustomPredictions instances

        Returns
        -------
        combined_predictions : list
            a single CustomPredictions instance

        """
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))

        # list of length N_predictions
        # each element in the list is a y_pred which is a numpy
        # array of length n_images. Each element in this array
        # is a list of predictions made for this image (for this model)
        y_pred_list = [predictions_list[i].y_pred for i in index_list]
        n_images = len(y_pred_list[0])

        all_predictions_by_image = [[] for _ in range(n_images)]
        num_predictions_by_image = [0 for _ in range(n_images)]
        for y_pred_for_model in y_pred_list:
            for image_index, predictions_for_image in enumerate(y_pred_for_model):
                if predictions_for_image is not None:
                    # predictions_for_image is a list of predictions
                    #   (each prediction is a dict {"class": xx, "proba": xx, "bbox": xx})
                    # that where made by a given model on a given image
                    all_predictions_by_image[image_index] += predictions_for_image
                    num_predictions_by_image[image_index] += 1

        # convert the result to a numpy array of list to make is compatible
        # with ramp indexing
        y_pred_combined = np.empty(n_images, dtype=object)
        y_pred_combined[:] = all_predictions_by_image
        # apply Non Maximum Suppression to remove duplicated predictions
        y_pred_combined = utils.apply_NMS_for_y_pred(
            y_pred_combined, iou_threshold=0.25
        )

        # we return a single CustomPredictions object with the combined predictions
        combined_predictions = cls(y_pred=y_pred_combined)
        return combined_predictions


# REQUIRED
Predictions = CustomPredictions
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
