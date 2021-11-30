import numpy as np

from rampwf.score_types.base import BaseScoreType
from .geometry import find_matching_bbox


class ClassAveragePrecision(BaseScoreType):
    """Compute average precision of predictions for one class.

    Example
    -------
    >>> X_train, y_train = problem.get_train_data()
    >>> y_pred = model.predict(X_train)
    >>> metric = ClassAveragePrecision(class_name="Secondary", iou_threshold=problem.SCORING_IOU)
    >>> metric(y_train, y_pred)
    0.823

    """

    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    worst = 0.0

    def __init__(self, class_name, iou_threshold):
        self.name = f"AP {class_name}"
        self.precision = 3

        self.class_name = class_name
        self.iou_threshold = iou_threshold

    def __call__(self, y_true, y_pred):

        precision, recall, _ = precision_recall_for_class(
            y_true, y_pred, self.class_name, self.iou_threshold
        )
        return average_precision(precision, recall)


class MeanAveragePrecision(BaseScoreType):
    """Compute mean of (average precision of predictions for one class).

    Example
    -------
    >>> X_train, y_train = problem.get_train_data()
    >>> y_pred = model.predict(X_train)
    >>> metric = ClassAveragePrecision(iou_threshold=problem.SCORING_IOU)
    >>> metric(y_train, y_pred)
    0.823

    """

    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    worst = 0.0

    def __init__(self, class_names, weights, iou_threshold):
        self.name = "mean AP"
        self.precision = 3

        self.class_names = class_names
        if weights is None:
            weights = [1 for _ in class_names]
        self.weights = weights
        self.iou_threshold = iou_threshold

    def __call__(self, y_true, y_pred):

        mean_AP = 0
        for class_name, weight in zip(self.class_names, self.weights):
            precision, recall, _ = precision_recall_for_class(
                y_true, y_pred, class_name, self.iou_threshold
            )
            mean_AP += weight * average_precision(precision, recall)
        mean_AP /= sum(self.weights)
        return mean_AP


def average_precision(precision, recall):
    """Compute average precision for a given precision/recall curve.

    definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score

    Warnings
    --------
    precision and recall arguments should be sorted by threshold.

    Returns
    -------
    average_precision : float

    """  # noqa : E501
    return float(
        sum(
            p * (r_i - r_i_1)
            for p, r_i, r_i_1 in zip(precision[1:], recall[1:], recall[:-1])
        )
    )


def precision_recall_for_class(y_true, y_pred, class_name, iou_threshold):
    """Compute precision/recall curve for a single class.

    Doc: https://github.com/rafaelpadilla/Object-Detection-Metrics

    Parameters
    ----------
    y_true : np.array (n_images,)
    y_pred : np.array (n_images,)
        output of model. Each element in the array is a list
        of predicted follicule locations for a single image.
        Each element in these lists are dicts following this structure ::

            {"bbox": (xmin, ymin, xmax, ymax), "category": "Primary", "proba": 0.872}

    class_name : str
        ex: 'Primary'
    iou_threshold : float
        Intersection Over Union threshold used to decide if two
        bounding boxes are the same.

    Returns
    -------
    precision : np.array
        all returned array have the same length. They are sorted by
        decreasing probability threshold.
    recall : np.array
    threshold : np.array

    """
    y_true = filter_class(y_true, class_name)
    y_pred = filter_class(y_pred, class_name)
    return _precision_recall_ignore_class(y_true, y_pred, iou_threshold)


def filter_class(y, class_name):
    """Filter a class in the output of a model

    Parameters
    ----------
    y : np.array
        array of predictions for multiple images. See :py:func:`precision_recall_for_class`
    class_name : str
        ex: 'Secondary'

    Returns
    -------
    filtered_y : np.array
        same shape as input y.

    '"""
    filtered = [
        [location for location in image_locations if location["class"] == class_name]
        for image_locations in y
    ]
    y_filtered = np.empty(len(filtered), dtype=object)
    y_filtered[:] = filtered
    return y_filtered


def _precision_recall_ignore_class(y_true, y_pred, iou_threshold):
    fake_image_names = [f"image_{i}" for i in range(len(y_true))]
    true_locations = []
    predicted_locations = []
    for image_name, true_locations_image, pred_locations_image in zip(
        fake_image_names, y_true, y_pred
    ):
        for true_loc in true_locations_image:
            true_locations.append({"image": image_name, **true_loc})
        for pred_loc in pred_locations_image:
            predicted_locations.append({"image": image_name, **pred_loc})

    predicted_locations = list(
        sorted(predicted_locations, key=lambda loc: loc["proba"], reverse=True)
    )

    precision = [1]
    recall = [0]
    threshold = [1]
    n_positive_detections = 0
    n_true_detected = 0
    n_true_to_detect = len(true_locations)
    for i, prediction in enumerate(predicted_locations):
        if len(true_locations) > 0:
            index, success = find_matching_bbox(
                prediction, true_locations, iou_threshold
            )
            if success:
                true_locations.pop(index)
                n_positive_detections += 1
                n_true_detected += 1

        threshold.append(prediction["proba"])
        precision.append(n_positive_detections / (i + 1))
        if n_true_to_detect > 0:
            recall.append(n_true_detected / n_true_to_detect)
        else:
            recall.append(0)

    return np.array(precision), np.array(recall), np.array(threshold)
