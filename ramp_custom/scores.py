import numpy as np

from rampwf.score_types.base import BaseScoreType


class ClassAveragePrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, class_name, iou_threshold=0.25):
        self.class_name = class_name
        self.iou_threshold = iou_threshold
        self.name = f"AP <{class_name}>"
        self.precision = 3

    def __call__(self, y_true, y_pred):

        precisions, recalls, thresholds = compute_precision_recall(y_true, y_pred)
        precision, recall = precisions[self.class_name], recalls[self.class_name]
        if False:
            print(
                f"Measuring AveragePrecision with inputs of size {len(y_true)} / {len(y_pred)}"
            )
            print(f"  computing averagePrecision from curve of {len(precision)} points")
        average_precision = sum(
            p * (r_i - r_i_1)
            for p, r_i, r_i_1 in zip(precision[1:], recall[1:], recall[:-1])
        )
        return average_precision


def compute_precision_recall(y_true, y_pred, iou_threshold=0.3):
    # STEP 1
    # Adatapt to previous way of doing things
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

    # print(true_locations)
    # STEP 2: precision / recall / thresholds for each class
    classes = [
        "Primordial",
        "Primary",
        "Secondary",
        "Tertiary",
    ]
    precisions = {}
    recalls = {}
    thresholds = {}
    for predicted_class in classes:
        true_boxes = [
            location
            for location in true_locations
            if location["class"] == predicted_class
        ]
        if not true_boxes:
            continue

        pred_boxes = [
            location
            for location in sorted(
                predicted_locations, key=lambda loc: loc["proba"], reverse=True
            )
            if location["class"] == predicted_class
        ]

        precision = [1]
        recall = [0]
        threshold = [1]
        n_positive_detections = 0
        n_true_detected = 0
        n_true_to_detect = len(true_boxes)
        for i, prediction in enumerate(pred_boxes):
            if len(true_boxes) > 0:
                index, success = find_matching_bbox(
                    prediction, true_boxes, iou_threshold
                )
                if success:
                    true_boxes.pop(index)
                    n_positive_detections += 1
                    n_true_detected += 1

            threshold.append(prediction["proba"])
            precision.append(n_positive_detections / (i + 1))
            recall.append(n_true_detected / n_true_to_detect)

        precisions[predicted_class] = precision
        recalls[predicted_class] = recall
        thresholds[predicted_class] = threshold

    return precisions, recalls, thresholds


def find_matching_bbox(prediction, list_of_true_values, iou_threshold):
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
    return index, maximum > iou_threshold


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
