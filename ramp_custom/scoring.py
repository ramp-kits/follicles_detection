import numpy as np

from rampwf.score_types.base import BaseScoreType


class ClassAveragePrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, class_name, iou_thresholdd=0.25):
        self.name = f"AP <{class_name}>"
        self.precision = 3

        self.class_name = class_name
        self.iou_thresholdd = iou_thresholdd

    def __call__(self, y_true, y_pred):

        precision, recall, _ = precision_recall_for_class(
            y_true, y_pred, self.class_name, self.iou_thresholdd
        )
        return average_precision(precision, recall)


class MeanAveragePrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, class_names, weights=None, iou_thresholdd=0.25):
        self.name = "mean AP"
        self.precision = 3

        self.class_names = class_names
        if weights is None:
            weights = [1 for _ in class_names]
        self.weights = weights
        self.iou_thresholdd = iou_thresholdd

    def __call__(self, y_true, y_pred):

        mean_AP = 0
        for class_name, weight in zip(self.class_names, self.weights):
            precision, recall, _ = precision_recall_for_class(
                y_true, y_pred, class_name, self.iou_thresholdd
            )
            mean_AP += weight * average_precision(precision, recall)
        mean_AP /= sum(self.weights)
        return mean_AP


def average_precision(precision, recall):
    """WARNING: expected to be sorted by threshold"""
    return sum(
        p * (r_i - r_i_1)
        for p, r_i, r_i_1 in zip(precision[1:], recall[1:], recall[:-1])
    )


def precision_recall_for_class(y_true, y_pred, class_name, iou_thresholdd):
    y_true = filter_class(y_true, class_name)
    y_pred = filter_class(y_pred, class_name)
    return precision_recall_ignore_class(y_true, y_pred, iou_thresholdd)


def filter_class(y, class_name):
    filtered = [
        [location for location in image_locations if location["class"] == class_name]
        for image_locations in y
    ]
    y_filtered = np.empty(len(filtered), dtype=object)
    y_filtered[:] = filtered
    return y_filtered


def precision_recall_ignore_class(y_true, y_pred, iou_thresholdd):
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
                prediction, true_locations, iou_thresholdd
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


def find_matching_bbox(prediction, list_of_true_values, iou_thresholdd):
    """

    Parameters
    ----------
    prediction: dict
        with keys "image", "bbox"
    list_of_true_values: list of dict
        same keys


    Return index, success
        index = index of bbox with highest iou
        success = if matching iou is greater than threshold
    """
    predicted_bbox = np.array(prediction["bbox"]).reshape(1, 4)
    all_true_bbox = np.array([value["bbox"] for value in list_of_true_values]).reshape(
        len(list_of_true_values), 4
    )

    ious = compute_iou(predicted_bbox, all_true_bbox)[0, :]
    is_different_image = np.array(
        [value["image"] != prediction["image"] for value in list_of_true_values]
    )
    ious[is_different_image] = 0

    index, maximum = np.argmax(ious), np.max(ious)
    return index, maximum > iou_thresholdd


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, x2, y2]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, xmax, ymax]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    lu = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rd = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = np.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return np.clip(intersection_area / union_area, 0.0, 1.0)


def apply_NMS_for_y_pred(y_pred, iou_threshold):
    filtered_predictions = [
        apply_NMS_for_image(predictions, iou_threshold) for predictions in y_pred
    ]
    y_pred_filtered = np.empty(len(y_pred), dtype=object)
    y_pred_filtered[:] = filtered_predictions
    return y_pred_filtered


def apply_NMS_for_image(predictions, iou_threshold):
    classes = set(pred["class"] for pred in predictions)
    filtered_predictions = []
    for class_name in classes:
        pred_for_class = [pred for pred in predictions if pred["class"] == class_name]
        filtered_pred_for_class = apply_NMS_ignore_class(pred_for_class, iou_threshold)
        filtered_predictions += filtered_pred_for_class
    return filtered_predictions


def apply_NMS_ignore_class(predictions, iou_threshold):
    selected_predictions = []
    predictions_sorted = list(
        sorted(predictions, key=lambda pred: pred["proba"], reverse=True)
    )
    while len(predictions_sorted) != 0:
        best_box = predictions_sorted.pop(0)
        selected_predictions.append(best_box)
        best_box_coords = np.array(best_box["bbox"]).reshape(1, -1)
        other_boxes_coords = np.array(
            [location["bbox"] for location in predictions_sorted]
        ).reshape(-1, 4)
        ious = compute_iou(best_box_coords, other_boxes_coords)
        for i, iou in reversed(list(enumerate(ious[0]))):
            if iou > iou_threshold:
                predictions_sorted.pop(i)
    return selected_predictions
