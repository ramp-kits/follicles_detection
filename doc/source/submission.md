# Creating your first submission

```python
import numpy as np
import random


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
            predictions_for_this_image = [
                {
                    "proba": random.random(),
                    "class": "Primordial",
                    "bbox": (1084, 5189, 1218, 5423),
                },
                {
                    "proba": random.random(),
                    "class": "Primordial",
                    "bbox": (2564, 3543, 2676, 3754),
                },
            ]
            predicted_locations.append(predictions_for_this_image)

        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = predicted_locations
        return y_pred

```

# Data structures

TODO