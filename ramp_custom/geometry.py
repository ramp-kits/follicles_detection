import numpy as np


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
