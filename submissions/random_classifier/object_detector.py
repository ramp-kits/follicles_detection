import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import problem
from ramp_custom.dev import do_profile


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[-1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


def predict_locations_for_windows(coupe, model, window_size=2000, num_windows=5000):
    boxes = list(
        generate_random_windows_for_image(
            coupe, window_size=window_size, num_windows=num_windows
        )
    )
    cropped_images = build_cropped_images(coupe, boxes, target_size=(224, 224))
    predicted_probas = model.predict(cropped_images)
    predicted_locations = convert_probas_to_locations(predicted_probas, boxes)
    return predicted_locations


def generate_random_windows_for_image(image_width, image_height, window_size, num_windows):
    """generator of square boxes that create a list of random
    windows of size ~ window_size for the given image"""
    assert len(window_size) == len(num_windows)
    all_boxes = []
    
    for size, n_boxes in zip(window_size, num_windows):
        mean = size
        std = 0.15 * size

        for _ in range(n_boxes):
            width = np.random.normal(mean, std)
            x1 = np.random.randint(0, image_width)
            y1 = np.random.randint(0, image_height)

            bbox = (x1, y1, x1 + width, y1 + width)
            all_boxes.append(bbox)
    return all_boxes


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
    index_to_class = {
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
                {"class": index_to_class[index], "proba": proba, "bbox": box}
            )
    return locations



class ObjectDetector:
    def __init__(self, internal_model_path=None):
        self.IMG_SHAPE = (224, 224, 3)  # 3-colored images

        if internal_model_path is not None:
            self._model = tf.keras.models.load_model(internal_model_path)
            return

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.IMG_SHAPE, include_top=False, weights="imagenet"
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=self.IMG_SHAPE)

        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(5, activation="softmax")

        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["sparse_categorical_accuracy"],
        )
        self._model = model

    def fit(self, X_image_paths, y_true_locations):
        """
        Parameters
        ----------
        X_image_paths : np.array of shape (N, )
            each element is a single absolute path to an image
        y_true_locations : np.array of shape (N, )
            each element is a list that represent the
            true locations of follicles in the corresponding image

        Returns
        -------
        self

        """
        # Our self._model takes as input a tensor (M, 224, 224, 3) and a class encoded as a number
        # Consequently we need to build these images from the files on disc
        class_to_index = {
            "Negative": 0,
            "Primordial": 1,
            "Primary": 2,
            "Secondary": 3,
            "Tertiary": 4,
        }

        thumbnails = []
        expected_predictions = []

        for filepath, locations in zip(X_image_paths, y_true_locations):
            print(f"reading {filepath}")
            image = problem.utils.load_image(filepath)

            for loc in locations:
                class_, bbox = loc["class"], loc["bbox"]

                prediction = class_to_index[class_]
                expected_predictions.append(prediction)

                thumbnail = image.crop(bbox)
                thumbnail = thumbnail.resize((224, 224))
                thumbnail = np.asarray(thumbnail)
                thumbnails.append(thumbnail)

        X_for_classifier = np.array(thumbnails)
        y_for_classifier = np.array(expected_predictions)
        self._model.fit(X_for_classifier, y_for_classifier, epochs=10)
        return self

    @do_profile()
    def predict(self, X):
        print("Running predictions ...")
        # X = numpy array N rows, 1 column, type object
        # each row = one file name for an image
        all_predictions = []
        for image_path in X:
            pred_list = self.predict_single_image(image_path)
            all_predictions.append(pred_list)

        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = all_predictions
        return y_pred

    @do_profile()
    def predict_single_image(self, image_path):
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_raw)
        width, height, depth = image.shape.as_list()
        boxes_sizes = [1000, 200]  # px
        boxes_amount = [200, 5_000]
        boxes = generate_random_windows_for_image(width, height, boxes_sizes, boxes_amount)

        images_tensor = [image]
        boxes_for_tf = [
            (y1 / height, x1 / width, y2 / height, x2 / width)
            for x1, y1, x2, y2 in boxes
        ]
        box_indices = [0] * len(boxes_for_tf)
        crop_size = (224, 224)
        cropped_images = tf.image.crop_and_resize(
            images_tensor, boxes_for_tf, box_indices, crop_size, method='bilinear',
            extrapolation_value=0, name=None
        )

        predicted_probas = self._model.predict(cropped_images)
        predicted_locations = convert_probas_to_locations(predicted_probas, boxes)
        return predicted_locations



def run_model():
    MODEL_PATH = os.path.join(problem.REPO_PATH, "models", "classifier")
    # Train model if it does not exist
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH, exist_ok=True)
        detector = ObjectDetector()
        X_train, y_train = problem.get_train_data()
        detector.fit(X_train, y_train)
        internal_model = detector._model
        internal_model.save(MODEL_PATH)

    # Load model from path
    detector = ObjectDetector(internal_model_path=MODEL_PATH)

    # # detector.fit(None, None)
    images_to_predict = ["D-1M06-3.jpg"]
    # # images_to_predict = ["D-1M01-3.jpg"]
    image_paths = [
        os.path.join(problem.REPO_PATH, "data", "test", ima)
        for ima in images_to_predict
    ]
    X = np.array(image_paths)
    predictions = detector.predict(X)

    # predictions = detector.predict_single_image()
    print(predictions)


@do_profile(follow=[build_cropped_images])
def prepare_image():
    image_path = os.path.join(problem.REPO_PATH, "data", "test", "D-1M06-3.jpg")
    image = problem.utils.load_image(image_path)
    boxes = list(
        generate_random_windows_for_image(
            image, window_size=1_000, num_windows=200
        )
    ) + list(
        generate_random_windows_for_image(
            image, window_size=200, num_windows=5_000
        )
    )
    cropped_images = build_cropped_images(image, boxes, target_size=(224, 224))
    infos = {
        "shape": cropped_images.shape,
        "memory" : int(cropped_images.nbytes / 1024 / 1024),
    }
    return infos


# @do_profile(follow=[])
def prepare_image_tensorflow():
    image_path = os.path.join(problem.REPO_PATH, "data", "test", "D-1M06-3.jpg")
    image_pil = problem.utils.load_image(image_path)
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_raw)

    boxes = list(
        generate_random_windows_for_image(
            image_pil, window_size=1_000, num_windows=200
        )
    ) + list(
        generate_random_windows_for_image(
            image_pil, window_size=200, num_windows=5_000
        )
    )

    width, height, depth = image.shape.as_list()

    images_tensor = [image]
    # boxes = [[0, 0, 0.1, 0.1]]
    boxes_for_tf = [
        (y1 / height, x1 / width, y2 / height, x2 / width)
        for x1, y1, x2, y2 in boxes
    ]
    box_indices = [0] * len(boxes_for_tf)
    crop_size = (224, 224)
    cropped_images = tf.image.crop_and_resize(
        images_tensor, boxes_for_tf, box_indices, crop_size, method='bilinear',
        extrapolation_value=0, name=None
    )

    infos = {
        "image shape": image.shape,
        "image dtype": image.dtype,
        "cropped images shape": cropped_images.shape,
        "cropped images dtype": cropped_images.dtype,
        # "memory" : int(cropped_images.nbytes / 1024 / 1024),
    }
    return cropped_images, boxes


if __name__ == "__main__":
    run_model()
    # infos = prepare_image()
    # infos = prepare_image_tensorflow()
    # print(infos)