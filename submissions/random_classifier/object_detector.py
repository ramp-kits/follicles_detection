"""
total runtime ~20min

----------------------------
Mean CV scores
----------------------------
	score AP <Primordial>    AP <Primary>  AP <Secondary>   AP <Tertiary>         mean AP          time
	train  0.001 ± 0.0008  0.047 ± 0.0353  0.308 ± 0.0659  0.398 ± 0.0362  0.188 ± 0.0136  93.7 ± 14.28
	valid  0.017 ± 0.0301  0.037 ± 0.0456  0.328 ± 0.1136  0.472 ± 0.1064  0.214 ± 0.0511  62.0 ± 11.18
	test   0.011 ± 0.0211  0.039 ± 0.0278  0.477 ± 0.1798    0.493 ± 0.17  0.255 ± 0.0404   20.4 ± 0.26
----------------------------
Bagged scores
----------------------------
	score  AP <Primordial>  AP <Primary>  AP <Secondary>  AP <Tertiary>  mean AP
	valid            0.004         0.013           0.376          0.625    0.254
	test             0.004         0.084           0.558          0.759    0.351


----------------------------
Mean CV scores
----------------------------
	score AP <Primordial>    AP <Primary>  AP <Secondary>   AP <Tertiary>         mean AP           time
	train    0.0 ± 0.0001  0.036 ± 0.0143  0.315 ± 0.0493  0.425 ± 0.0558  0.194 ± 0.0235  104.3 ± 13.93
	valid   0.003 ± 0.005  0.041 ± 0.0632  0.336 ± 0.1355  0.514 ± 0.0624  0.224 ± 0.0454   121.7 ± 1.43
	test   0.001 ± 0.0021  0.009 ± 0.0117  0.471 ± 0.1009  0.297 ± 0.1463  0.194 ± 0.0227    21.5 ± 0.19
----------------------------
Bagged scores
----------------------------
	score  AP <Primordial>  AP <Primary>  AP <Secondary>  AP <Tertiary>  mean AP
	valid            0.000         0.014           0.390          0.646    0.263
	test             0.001         0.018           0.749          0.690    0.365
    
"""
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

def load_tf_image(image_path):
    from tensorflow.python.framework.errors_impl import InvalidArgumentError
    try:
        image_raw = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_raw)
    except InvalidArgumentError:
        # image is too large
        image_pil = problem.utils.load_image(image_path)
        image_np = np.asarray(image_pil)
        image = tf.convert_to_tensor(image_np)
    return image


def generate_random_windows_for_image(image, window_size, num_windows):
    """generator of square boxes that create a list of random
    windows of size ~ window_size for the given image"""
    assert len(window_size) == len(num_windows)
    image_height, image_width, _ = image.shape.as_list()
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


def build_cropped_images(image, boxes, crop_size):
    """Crop subimages in large image and resize them to a single size.

    Parameters
    ----------
    image: tf.Tensor shape (height, width, depth)
    boxes: list of tuple
        each element in the list is (xmin, ymin, xmax, ymax)
    crop_size : tuple(2)
        size of the returned cropped images
        ex: (224, 224)

    Returns
    -------
    cropped_images : np.array
        example shape (N_boxes, 224, 224, 3)

    """
    height, width, _ = image.shape.as_list()

    images_tensor = [image]
    boxes_for_tf = [
        (y1 / height, x1 / width, y2 / height, x2 / width)
        for x1, y1, x2, y2 in boxes
    ]
    box_indices = [0] * len(boxes_for_tf)
    cropped_images = tf.image.crop_and_resize(
        images_tensor, boxes_for_tf, box_indices, crop_size, method='bilinear',
        extrapolation_value=0, name=None
    )
    return cropped_images


def predict_single_image(image_path, model):
        image = load_tf_image(image_path)
        

        # width, height, depth = image.shape.as_list()
        boxes_sizes = [3000, 1000, 300]  # px
        boxes_amount = [200, 500, 2_000]
        boxes = generate_random_windows_for_image(image, boxes_sizes, boxes_amount)
        cropped_images = build_cropped_images(image, boxes, (224, 224))

        predicted_probas = model.predict(cropped_images)
        predicted_locations = convert_probas_to_locations(predicted_probas, boxes)
        return predicted_locations

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

    @do_profile()
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

            expected_preds_for_image = [
                class_to_index[loc["class"]]
                for loc in locations
            ]
            expected_predictions += expected_preds_for_image

            image = load_tf_image(filepath)
            
            boxes = [loc["bbox"] for loc in locations]
            thumbnails_for_image = build_cropped_images(image, boxes, self.IMG_SHAPE[0:2])
            thumbnails.append(thumbnails_for_image)
            

        X_for_classifier = tf.concat(thumbnails, axis=0)
        y_for_classifier = tf.constant(expected_predictions)
        infos = {
            "X": X_for_classifier.shape,
            "y": y_for_classifier.shape
        }
        print(infos)
        self._model.fit(X_for_classifier, y_for_classifier, epochs=100)
        return self

    

    @do_profile(follow=[predict_single_image])
    def predict(self, X):
        print("Running predictions ...")
        # X = numpy array N rows, 1 column, type object
        # each row = one file name for an image
        y_pred = np.empty(len(X), dtype=object)
        for i, image_path in enumerate(X):
            prediction_list_for_image = predict_single_image(image_path, self._model)
            y_pred[i] = prediction_list_for_image

        return y_pred

    




def run_model():
    MODEL_PATH = os.path.join(problem.REPO_PATH, "models", "classifier")
    DO_TRAIN = True
    # Train model if it does not exist
    if DO_TRAIN or not os.path.exists(MODEL_PATH):
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



if __name__ == "__main__":
    run_model()
    # infos = prepare_image()
    # infos = prepare_image_tensorflow()
    # print(infos)