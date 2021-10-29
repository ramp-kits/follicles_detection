import numpy as np
from rampwf.prediction_types.detection import Predictions as DetectionPredictions

from .geometry import apply_NMS_for_y_pred


def make_custom_predictions(iou_threshold):
    """Create class CustomPredictions using iou_threshold when bagging."""

    class CustomPredictions(DetectionPredictions):
        @classmethod
        def combine(cls, predictions_list, index_list=None):
            """Combine multiple predictions into a single one.

            This is used when the "bagged scores" are computed.

            Parameters
            ----------
            predictions_list : list
                list of CustomPredictions instances

            Returns
            -------
            combined_predictions : list
                a single CustomPredictions instance

            """
            if index_list is None:  # we combine the full list
                index_list = range(len(predictions_list))

            # list of length N_predictions
            # each element in the list is a y_pred which is a numpy
            # array of length n_images. Each element in this array
            # is a list of predictions made for this image (for this model)
            y_pred_list = [predictions_list[i].y_pred for i in index_list]
            n_images = len(y_pred_list[0])

            all_predictions_by_image = [[] for _ in range(n_images)]
            num_predictions_by_image = [0 for _ in range(n_images)]
            for y_pred_for_model in y_pred_list:
                for image_index, predictions_for_image in enumerate(y_pred_for_model):
                    if predictions_for_image is not None:
                        # predictions_for_image is a list of predictions
                        #   (each prediction is a dict {"class": xx, "proba": xx, "bbox": xx})
                        # that where made by a given model on a given image
                        all_predictions_by_image[image_index] += predictions_for_image
                        num_predictions_by_image[image_index] += 1

            # convert the result to a numpy array of list to make is compatible
            # with ramp indexing
            y_pred_combined = np.empty(n_images, dtype=object)
            y_pred_combined[:] = all_predictions_by_image
            # apply Non Maximum Suppression to remove duplicated predictions
            y_pred_combined = apply_NMS_for_y_pred(
                y_pred_combined, iou_threshold=iou_threshold
            )

            # we return a single CustomPredictions object with the combined predictions
            combined_predictions = cls(y_pred=y_pred_combined)
            return combined_predictions

    return CustomPredictions
