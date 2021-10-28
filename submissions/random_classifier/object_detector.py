import os
import numpy as np
import tensorflow as tf


import sys

repo_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(repo_folder)
from ramp_custom.utils import load_image
from ramp_custom.utils import do_profile

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

    # @do_profile(follow=[predict_locations_for_windows, build_cropped_images])
    def predict(self, X):
        print("Running predictions ...")
        # X = numpy array N rows, 1 column, type object
        # each row = one file name for an image
        all_predictions = []
        for i, image_path in enumerate(X):
            # TEMP: only make prediction for first image
            if i == 0:
                img = load_image(image_path)
                pred_list = predict_locations_for_windows(img, self._model)
                all_predictions.append(pred_list)
            else:
                all_predictions.append([])

        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = all_predictions
        return y_pred


def predict_locations_for_windows(coupe, model, window_size=1000, num_windows=10):
    boxes = list(
        generate_random_windows_for_image(
            coupe, window_size=window_size, num_windows=num_windows
        )
    )
    cropped_images = build_cropped_images(coupe, boxes, target_size=(224, 224))
    predicted_probas = model.predict(cropped_images)
    predicted_locations = convert_probas_to_locations(predicted_probas, boxes)
    return predicted_locations


def generate_random_windows_for_image(image, window_size, num_windows):
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


def build_cropped_images(image, boxes, target_size):
    """Crop subimages in large image and resize them to a single size.

    Parameters
    ----------
    image: PIL.Image
    boxes: list of tuple
        each element in the list is (xmin, ymin, xmax, ymax)
    target_size : tuple(2)
        size of the returned cropped images
        ex: (224, 224)

    Returns
    -------
    cropped_images : np.array
        example shape (N_boxes, 224, 224, 3)

    """
    cropped_images = []
    for box in boxes:
        cropped_image = image.crop(box)
        cropped_image = cropped_image.resize(target_size)
        cropped_image = np.array(cropped_image)
        cropped_images.append(cropped_image)
    return np.array(cropped_images)


def convert_probas_to_locations(probas, boxes):
    top_index, top_proba = np.argmax(probas, axis=1), np.max(probas, axis=1)
    index_to_label = {
        0: "Negative",
        1: "Primordial",
        2: "Primary",
        3: "Secondary",
        4: "Tertiary",
    }
    locations = []
    for index, proba, box in zip(top_index, top_proba, boxes):
        if index != 0:
            locations.append(
                {"label": index_to_label[index], "proba": proba, "bbox": box}
            )
    return locations


if __name__ == "__main__":
    detector = ObjectDetector()
    # detector.fit(None, None)
    images_to_predict = ["D-1M06-3.jpg"]
    # images_to_predict = ["D-1M01-3.jpg"]
    image_paths = [
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "coupes_jpg", ima
            )
        )
        for ima in images_to_predict
    ]
    X = np.array(image_paths)  # , "D-1M01-4.jpg"])
    predictions = detector.predict(X)
    print(predictions)
