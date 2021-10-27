import os
import numpy as np
from PIL import Image
import tensorflow as tf

this_folder = os.path.abspath("")
DATA_FOLDER = os.path.join(this_folder, "data")
MODELS_FOLDER = os.path.join(this_folder, "models")
MODEL_NAME_FOR_PREDICTION = "classifier2"



class ObjectDetector:
    def __init__(self):
        pass

    def fit(self, X, y):
        # ignore inputs
        # load trained model from file
        self._model = tf.keras.models.load_model(
            os.path.join(MODELS_FOLDER, MODEL_NAME_FOR_PREDICTION)
        )
        return self

    def predict_image(image, model):
        category_to_label = {
            0: "Negative",
            1: "Primordial", 
            2: "Primary", 
            3: "Secondary", 
            4: "Tertiary", 
        }

        image = image.resize((224,224))
        image = np.array(image)
        image = tf.reshape(image, (1,224,224,3))
        pred = model.predict(image)
        predicted_category, proba = np.argmax(pred),  np.max(pred)
        predicted_label = category_to_label[predicted_category]
        return predicted_label, proba

    
    def generate_random_windows_for_image(self, image, window_size=1000, num_windows=1000):
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

    def predict_locations_for_windows(self, coupe, window_size, model, num_windows=1000):
        boxes = generate_random_windows_for_image(self, coupe, window_size=window_size, num_windows=num_windows)
        predicted_locations = []
        for box in boxes:
            cropped_image = coupe.crop(box)
            label, proba = predict_image(self, cropped_image, model)
            predicted_locations.append({
                    "bbox": box,
                    "label": label,
                    "proba": proba
                })
        return predicted_locations

    def predict(self, X):
        # X = numpy array N rows, 1 column, type object
         # each row = one file name for an image
        all_predictions = []
        for img_name in X:
            img = Image.open(os.path.join(DATA_FOLDER, "coupes_jpg", img_name))
            pred_list = self.predict_locations_for_windows(self, img)
        all_predictions.append(pred_list)
        return all_predictions