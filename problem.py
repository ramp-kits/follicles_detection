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
from rampwf.prediction_types.detection import Predictions as DetectionPredictions
from rampwf.score_types import Combined
from sklearn.model_selection import LeaveOneGroupOut

sys.path.append(os.path.dirname(__file__))
from ramp_custom.scores import ClassAveragePrecision, MeanAveragePrecision

problem_title = "Follicle Detection and Classification"


class CustomPredictions(DetectionPredictions):
    @classmethod
    def combine(cls, predictions_list, index_list=None, greedy=False):
        # if index_list is None:  # we combine the full list
        #     index_list = range(len(predictions_list))
        # y_comb_list = np.array([predictions_list[i].y_pred for i in index_list])
        # combined_predictions = cls(y_pred=y_comb_list)
        if False:
            print("Combine!")
            print(
                f"Calling combine with a list of {len(predictions_list)} Predictions object"
            )
            print("index is none : ", index_list is None)
            for p in predictions_list:
                print(f"     - len(y) :  {len(p.y_pred)}")

        return predictions_list[0]


# REQUIRED
Predictions = CustomPredictions
workflow = ObjectDetector()
average_precisions = [
    ClassAveragePrecision("Primordial"),
    ClassAveragePrecision("Primary"),
    ClassAveragePrecision("Secondary"),
    ClassAveragePrecision("Tertiary"),
    MeanAveragePrecision(
        class_names=["Primordial", "Primary", "Secondary", "Tertiary"]
    ),
]

score_types = [
    *average_precisions,
    # Combined(
    #     name="Mean AP",
    #     score_types=average_precisions,
    #     weights=[1, 1, 1, 1],
    #     precision=3,
    # ),
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
