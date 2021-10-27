import numpy as np


class ObjectDetector:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        predicted_locations = []
        for image_path in X:
            # image = load_image(image_path)
            # etc.
            x, y, width = 10, 50, 200
            predictions_for_this_image = [
                {"label": "Secondary", "bbox": (x, y, x + 2 * width, y + 2 * width)},
                {"label": "Primary", "bbox": (x, y, x + width, y + width)},
                {"label": "Primordial", "bbox": (x, y, x + width, y + width)},
            ]
            predicted_locations.append(predictions_for_this_image)

        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = predicted_locations
        return y_pred
