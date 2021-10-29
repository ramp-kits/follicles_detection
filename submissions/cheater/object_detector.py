import os
import numpy as np
import random
import problem


class ObjectDetector:
    """Dummy object detector used to verify that we compute metrics accurately

    It detects perfecly on train set except that:
        - it detects nothing on the first CV fold
        - it detects nothing for class Primary
        - it detects 50% of class Primordial
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        wrong_location = {"class": "Primordial", "proba": 0.5, "bbox": (0, 0, 10, 10)}
        if os.environ.get("IS_FIRST_FOLD", "true") == "true":
            os.environ["IS_FIRST_FOLD"] = "false"
            pred = [[wrong_location] for _ in X]
        else:
            x_train, y_train = problem.get_train_data()
            path_to_y = {path: y for path, y in zip(x_train, y_train)}

            pred = [path_to_y.get(path, [wrong_location]) for path in X]

            def keep_prediction(location):
                if location["class"] == "Primary":
                    return False
                if location["class"] == "Primordial":
                    return random.random() > 0.5
                return True

            pred = [
                [
                    {"proba": random.random(), **location}
                    for location in image_locations
                    if keep_prediction(location)
                ]
                for image_locations in pred
            ]

        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = pred
        return y_pred
