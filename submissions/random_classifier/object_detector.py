"""
Result of `ramp-test --submision random_classifier`

total runtime ~30min

----------------------------
Mean CV scores
----------------------------
score AP <Primordial>    AP <Primary>  AP <Secondary>   AP <Tertiary>         mean AP           time
train  0.001 ± 0.0007  0.023 ± 0.0077   0.375 ± 0.025  0.436 ± 0.0563  0.209 ± 0.0208  130.8 ± 20.49
valid       0.0 ± 0.0  0.037 ± 0.0214  0.332 ± 0.0734  0.458 ± 0.0557  0.207 ± 0.0107   166.2 ± 5.91
test   0.001 ± 0.0018  0.008 ± 0.0141  0.325 ± 0.1782  0.398 ± 0.1259  0.183 ± 0.0683    32.3 ± 0.48
----------------------------
Bagged scores
----------------------------
score  AP <Primordial>  AP <Primary>  AP <Secondary>  AP <Tertiary>  mean AP
valid            0.000         0.026           0.361          0.626    0.253
test             0.002         0.008           0.475          0.625    0.278
"""
import numpy as np
import tensorflow as tf
from matplotlib.image import imread
import PIL

PIL.Image.MAX_IMAGE_PIXELS = None  # otherwise large images cannot be loaded

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the last GPU
    try:
        tf.config.set_visible_devices(gpus[-1], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


class ObjectDetector:
    def __init__(self):
        self.IMG_SHAPE = (
            224,
            224,
            3,
        )  # size of thumbnails used by the internal classification model
        self.CLASS_TO_INDEX = {
            "Negative": 0,
            "Primordial": 1,
            "Primary": 2,
            "Secondary": 3,
            "Tertiary": 4,
        }  # how to convert the provided classes of follicules to numbers
        self.INDEX_TO_CLASS = {value: key for key, value in self.CLASS_TO_INDEX.items()}

        # create classifier model based on MobileNet
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
        # We need to build this tensor of small images (thumbnails) from the data

        thumbnails = []
        expected_predictions = []

        for filepath, locations in zip(X_image_paths, y_true_locations):

            boxes_for_images = []
            for true_location in locations:
                expected_class = true_location["class"]
                true_bbox = true_location["bbox"]

                expected_pred = self.CLASS_TO_INDEX[expected_class]
                expected_predictions.append(expected_pred)

                boxes_for_images.append(true_bbox)

            image = imread(filepath)
            thumbnails_for_image = build_cropped_images(
                image, boxes_for_images, self.IMG_SHAPE[0:2]
            )
            thumbnails.append(thumbnails_for_image)

        X_for_classifier = tf.concat(
            thumbnails, axis=0
        )  # thumbnails as tensor of shape (N_boxes, 224, 224, 3)
        y_for_classifier = tf.constant(
            expected_predictions
        )  # class to predict as tensor of shape (N_boxes)
        self._model.fit(X_for_classifier, y_for_classifier, epochs=100)
        return self

    def predict(self, X):
        print("Running predictions ...")
        # X = numpy array N rows, 1 column, type object
        # each row = one file name for an image
        y_pred = np.empty(len(X), dtype=object)
        for i, image_path in enumerate(X):
            prediction_list_for_image = self.predict_single_image(image_path)
            y_pred[i] = prediction_list_for_image

        return y_pred

    def predict_single_image(self, image_path):
        image = imread(image_path)

        boxes_sizes = [3000, 1000, 300]  # px
        boxes_amount = [200, 500, 2_000]
        boxes = generate_random_windows_for_image(image, boxes_sizes, boxes_amount)
        cropped_images = build_cropped_images(
            image, boxes, crop_size=self.IMG_SHAPE[0:2]
        )

        predicted_probas = self._model.predict(cropped_images)
        predicted_locations = self.convert_probas_to_locations(predicted_probas, boxes)
        return predicted_locations

    def convert_probas_to_locations(self, probas, boxes):
        top_index, top_proba = np.argmax(probas, axis=1), np.max(probas, axis=1)
        predicted_locations = []
        for index, proba, box in zip(top_index, top_proba, boxes):
            if index != 0:
                predicted_locations.append(
                    {"class": self.INDEX_TO_CLASS[index], "proba": proba, "bbox": box}
                )
        return predicted_locations


def generate_random_windows_for_image(image, window_sizes, num_windows):
    """create list of bounding boxes of varying sizes

    Parameters
    ----------
    image : np.array
    window_sizes : list of int
        exemple [200, 1000, 2000]
        sizes of windows to use
    num_windows : list of int
        example [1000, 100, 100]
        how many boxes of each window_size should be created ?

    """
    assert len(window_sizes) == len(num_windows)
    image_height, image_width, _ = image.shape
    all_boxes = []

    for size, n_boxes in zip(window_sizes, num_windows):
        mean = size
        std = 0.15 * size

        for _ in range(n_boxes):
            width = np.random.normal(mean, std)
            x1 = np.random.randint(0, image_width)
            y1 = np.random.randint(0, image_height)

            bbox = (x1, y1, x1 + width, y1 + width)
            all_boxes.append(bbox)
    return all_boxes


def build_cropped_images(image, boxes, crop_size):
    """Crop subimages in large image and resize them to a single size.

    Parameters
    ----------
    image : np.array of shape (height, width, depth)
    boxes : list of tuple
        each element in the list is (xmin, ymin, xmax, ymax)
    crop_size : tuple(2)
        size of the returned cropped images
        ex: (224, 224)

    Returns
    -------
    cropped_images : tensor
        example shape (N_boxes, 224, 224, 3)

    """
    height, width, _ = image.shape
    images_tensor = [tf.convert_to_tensor(image)]
    # WARNING: tf.convert_to_tensor([image])   does not seem to work..
    boxes_for_tf = [
        (y1 / height, x1 / width, y2 / height, x2 / width) for x1, y1, x2, y2 in boxes
    ]
    box_indices = [0] * len(boxes_for_tf)
    cropped_images = tf.image.crop_and_resize(
        images_tensor,
        boxes_for_tf,
        box_indices,
        crop_size,
        method="bilinear",
        extrapolation_value=0,
        name=None,
    )
    return cropped_images


if __name__ == "__main__":
    print("This should be run with 'ramp-test --submission random_classifier'")
