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
import os
import pandas as pd

from rampwf.workflows import ObjectDetector
from rampwf.prediction_types.base import BasePrediction

from sklearn.model_selection import LeaveOneGroupOut


from ramp_custom.scores import AveragePrecision

problem_title = "Follicle Detection and Classification"

# A type (class) which will be used to create wrapper objects for y_pred
# Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
Predictions = BasePrediction

# An object implementing the workflow
workflow = ObjectDetector()

score_types = [AveragePrecision()]


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


def format_labels_to_problem_data(df):
    """
    df: pd.DataFrame
        columns: image, label
                  xmin, ymin, xmax, ymax
    """
    X = df["image"]
    y = [
        {
            "label": row["label"],
            "bbox": (row["xmin"], row["ymin"], row["xmax"], row["ymax"]),
        }
        for row in df.iterrows()
    ]
    return X, y


def get_train_data(path="."):
    labels = pd.read_csv(os.path.join(path, "data", "labels.csv"))
    labels = labels.loc[labels["set"] == "train"]
    return format_labels_to_problem_data(labels)


def get_test_data(path="."):
    labels = pd.read_csv(os.path.join(path, "data", "labels.csv"))
    labels = labels.loc[labels["set"] == "test"]
    return format_labels_to_problem_data(labels)
