import os
import numpy as np
import random
import problem


class ObjectDetector:
    """Dummy object detector used to verify that we compute metrics accurately

    It detects perfecly on train set and test set except that:
        - it detects nothing for class Primary
        - it detects randomly 50% of class Primordial
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        x_train, y_train = problem.get_train_data()
        x_test, y_test = problem.get_test_data()
        path_to_y = {
            **{path: y for path, y in zip(x_train, y_train)},
            **{path: y for path, y in zip(x_test, y_test)},
        }

        pred = [path_to_y.get(path, []) for path in X]

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
