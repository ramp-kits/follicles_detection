import os
import numpy as np
import tensorflow as tf


import sys

repo_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(repo_folder)
from ramp_custom.utils import load_image

# IMAGES_FOLDER = os.path.join(repo_folder, "data", "coupes_jpg")
MODELS_FOLDER = os.path.join(repo_folder, "models")
MODEL_NAME_FOR_PREDICTION = "classifier"


class ObjectDetector:
    def __init__(self):
        self._model = tf.keras.models.load_model(
            os.path.join(MODELS_FOLDER, MODEL_NAME_FOR_PREDICTION)
        )

    def fit(self, X, y):
        # ignore inputs for now
        return self

    def predict(self, X):
        print("Running predictions ...")
        # X = numpy array N rows, 1 column, type object
        # each row = one file name for an image
        all_predictions = []
        for i, image_path in enumerate(X):
            # TEMP: only make prediction for first image
            if i == 0:
                print(f"   prediction for image {image_path}")
                img = load_image(image_path)
                pred_list = self.predict_locations_for_windows(img, self._model)
                all_predictions.append(pred_list)
            else:
                all_predictions.append([])

        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = all_predictions
        return y_pred

    def predict_image(self, image, model):
        category_to_label = {
            0: "Negative",
            1: "Primordial",
            2: "Primary",
            3: "Secondary",
            4: "Tertiary",
        }

        image = image.resize((224, 224))
        image = np.array(image)
        image = tf.reshape(image, (1, 224, 224, 3))
        pred = model.predict(image)
        predicted_category, proba = np.argmax(pred), np.max(pred)
        predicted_label = category_to_label[predicted_category]
        return predicted_label, proba

    def generate_random_windows_for_image(self, image, window_size, num_windows):
        """generator of square boxes that create a list of random
        windows of size ~ window_size for the given image"""
        mean = window_size
        std = 0.15 * window_size

        c = 0
        while True:
            width = np.random.normal(mean, std)
            x1 = np.random.randint(0, image.width)
            y1 = np.random.randint(0, image.height)

            bbox = (x1, y1, x1 + width, y1 + width)
            yield bbox
            c += 1

            if c > num_windows:
                break

    def predict_locations_for_windows(
        self, coupe, model, window_size=1000, num_windows=10
    ):
        boxes = self.generate_random_windows_for_image(
            coupe, window_size=window_size, num_windows=num_windows
        )
        predicted_locations = []
        for box in boxes:
            cropped_image = coupe.crop(box)
            label, proba = self.predict_image(cropped_image, model)
            if label != "Negative":
                predicted_locations.append(
                    {"bbox": box, "label": label, "proba": proba}
                )
        return predicted_locations


if __name__ == "__main__":
    detector = ObjectDetector()
    detector.fit(None, None)
    X = np.array(["D-1M01-3.jpg", "D-1M01-4.jpg"])
    predictions = detector.predict(X)
    print(predictions)
