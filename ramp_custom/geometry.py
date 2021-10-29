import numpy as np


def find_matching_bbox(prediction, list_of_true_values, iou_threshold):
    """Find the index of the bounding box that is closest to the prediction.

    Parameters
    ----------
    prediction : dict
        with keys "image", "bbox". This is the reference bounding box against which
        we are comparing.
    list_of_true_values : list of dict
        same keys as prediction. Only true_values belonging to the same image
        as the prediction are taken into account.


    Returns
    -------
    index : int
        index in the ``list_of_true_values`` array of the bounding box that
        is the closest to the ``prediction``.
    success : bool
        True if the IoU between the prediction and highest scoring true_value
        is higher than the ``iou_threshold``

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
    return index, maximum > iou_threshold


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    IOU : Intersection over Union.
    0 < IOU < 1

    Doc: https://github.com/rafaelpadilla/Object-Detection-Metrics

    Parameters
    ----------
    boxes1 : np.array of shape `(N, 4)`
        these representi bounding boxes
        where each box is of the format `[x, y, x2, y2]`.
    boxes2 : np.array shape `(M, 4)`
        same format as ``boxes1``

    Returns
    -------
    iou_matrix : np.array of shape `(N, M)`
        pairwise IOU matrix with shape `(N, M)`, where the value at i_th row
        j_th column holds the IOU between ith box and jth box from
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
    """Remove duplicate predictions using Non Negative Suppression.

    Parameters
    ----------
    y_pred : np.array of shape (N_images,)
        each element is a list of predicted locations.
    iou_threshold : float
        if multiple locations (of the same class and for the same image)
        have an IoU higher than this threshold, only the one with the highest
        probability is kept.

    Returns
    -------
    y_pred_filtered : np.array of shape (N_images,)

    """
    filtered_predictions = [
        apply_NMS_for_image(predictions, iou_threshold) for predictions in y_pred
    ]
    y_pred_filtered = np.empty(len(y_pred), dtype=object)
    y_pred_filtered[:] = filtered_predictions
    return y_pred_filtered


def apply_NMS_for_image(predictions, iou_threshold):
    """Remove duplicate predictions using Non Negative Suppression.

    Similar to ``apply_NMS_for_y_pred`` but expects a list of
    predictions as input.

    """
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
