"""
Doc:
- https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/problem.html


Examples of problem.py:
- basic titanic: https://github.com/ramp-kits/titanic/blob/master/problem.py
- for mars challenge: https://github.com/ramp-kits/mars_craters/blob/master/problem.py
- where everything is custom: https://github.com/ramp-kits/meg/blob/master/problem.py


What we need to define:

1. Prediction type
    - probably cannot use make_detection() as it uses a function greedy_nms
        https://github.com/paris-saclay-cds/ramp-workflow/blob/212720ff677985f57a0f26e073df9bad6dc5c9c0/rampwf/prediction_types/detection.py#L84
      that rely on the computation of IoU between two circles.
      Note: this method is only called in the `combine()` method of the Predictions class.
      Maybe we can use this if we do not rely on `combine()`.

    - custom problem implements a class _MultiOutputClassification(BasePrediction)

2. Workflow
    -> c'est ça qui va chercher la submission et qui la lance
    - peut être qu'on peut utiliser le `Estimator()` de base ?
    - je pense qu'on peut utiliser le ObjectDetector() 
      https://github.com/paris-saclay-cds/ramp-workflow/blob/212720ff677985f57a0f26e073df9bad6dc5c9c0/rampwf/workflows/object_detector.py#L10
      qui a l'air assez simple dans son fonctionnement.


3. Des fonctions de score
    - on ne peut pas utiliser celle utilisées par le mars challenge
    (ex     rw.score_types.DetectionAveragePrecision(name='ap'),)
    car elles se basent sur des cercles sans catégorie



"""
import re
import sys
import os
import pandas as pd
import numpy as np

from rampwf.workflows import ObjectDetector

# from rampwf.prediction_types.base import BasePrediction
# from rampwf.prediction_types import make_detection
from rampwf.prediction_types.detection import (
    Predictions as DetectionPredictions,
)
from sklearn.model_selection import LeaveOneGroupOut

sys.path.append(os.path.dirname(__file__))
from ramp_custom.scores import (
    ClassAveragePrecision,
    MeanAveragePrecision,
    apply_NMS_for_y_pred,
)

problem_title = "Follicle Detection and Classification"


class CustomPredictions(DetectionPredictions):
    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """
        Parameters:
            predictions_list : list of CustomPredictions instances

        Returns:
            combined_predictions : a single CustomPredictions instance

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
        y_pred_combined = apply_NMS_for_y_pred(y_pred_combined, iou_threshold=0.25)

        # we return a single CustomPredictions object with the combined predictions
        combined_predictions = cls(y_pred=y_pred_combined)
        return combined_predictions


# REQUIRED
Predictions = CustomPredictions
workflow = ObjectDetector()
score_types = [
    ClassAveragePrecision("Primordial"),
    ClassAveragePrecision("Primary"),
    ClassAveragePrecision("Secondary"),
    ClassAveragePrecision("Tertiary"),
    MeanAveragePrecision(
        class_names=["Primordial", "Primary", "Secondary", "Tertiary"]
    ),
]


def get_cv(X, y):
    """
    X: list of image names
    """

    def extract_ovary_number(filename):
        digit = re.match(r".*M0(\d)-\d.*", filename).group(1)
        return int(digit)

    groups = [extract_ovary_number(filename) for filename in X]
    cv = LeaveOneGroupOut()
    return cv.split(X, y, groups)


def _get_data(path=".", split="train"):
    """
    return: X: array of N image paths
            y: array of N lists of dicts: {"bbox", "class"}
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
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")
