import os
import numpy as np
import random
import problem


class ObjectDetector:
    """Dummy object detector used to verify that we compute metrics accurately

    It detects perfecly on train set for all classes except Primordial
    It detects nothing on the first CV fold
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        if os.environ.get("IS_FIRST_FOLD", "true") == "true":
            os.environ["IS_FIRST_FOLD"] = "false"
            pred = [
                [{"class": "Primordial", "proba": 0.5, "bbox": (0, 0, 10, 10)}]
                for _ in X
            ]
        else:
            x_train, y_train = problem.get_train_data()
            path_to_y = {path: y for path, y in zip(x_train, y_train)}

            pred = [path_to_y.get(path, []) for path in X]
            pred = [
                [
                    {"proba": random.random(), **location}
                    for location in image_locations
                    if location["class"] != "Primary"
                ]
                for image_locations in pred
            ]

        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = pred
        return y_pred
