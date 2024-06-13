import os
import pickle
from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import io
import base64
import sklearn
from tensorflow.keras.applications.vgg16 import  preprocess_input


app = Flask(__name__)

# Define the dice_coef function
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Loading pre-trained models
brain_breast_classifier = keras.models.load_model('models/Brain_Breast_Classifier.h5')
brain_tumor_classifier = keras.models.load_model('models/brain_classifier.h5')
brain_unet = keras.models.load_model('models/BRAIN-UNET-320epochs.h5', custom_objects={'dice_coef': dice_coef})
breast_tumor_classifier = keras.models.load_model('models/Breast_cancer_classification_model.h5')
breast_unet = keras.models.load_model('models/Breast_Cancer_Segment.h5', custom_objects={'dice_coef': dice_coef})

with open('models/one_class_svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)
vg16 = keras.models.load_model('models/vg16.h5')

def svm_predict(image):
    # Resize and preprocess the image for SVM prediction
    image_resized = cv2.resize(image, (224, 224))
    image_array = image_resized.reshape(-1, 224, 224, 3)
    preprocess_image= preprocess_input(np.array(image_array))
    feature = vg16.predict(preprocess_image)
    feature_flattened = feature.flatten().reshape(1, -1)
    prediction = svm_model.predict(feature_flattened)
    return prediction  # 1 for known, -1 for unknown

def predict_scan_type(image):
    image_resized = cv2.resize(image, (128, 128))
    prediction = brain_breast_classifier.predict(image_resized.reshape(-1, 128, 128, 3), verbose=0)
    return int(prediction)  # 0 for brain scan, 1 for breast scan

def predict_tumor(image):
    image_resized = cv2.resize(image, (128, 128))
    prediction = brain_tumor_classifier.predict(image_resized.reshape(-1, 128, 128, 3), verbose=0)
    return int(prediction)

def predict_mask(image):
    image_resized = cv2.resize(image, (128, 128))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY).reshape(128, 128, 1)
    mask_pred = np.squeeze(brain_unet.predict(np.expand_dims(image_gray, axis=0), verbose=0), axis=0)
    return mask_pred  # Return the raw prediction probabilities for the heatmap

def predict_breast_mask(image):
    image_resized = cv2.resize(image, (128, 128))
    image_gray = image_resized[:,:,0].reshape(128,128,1)
    mask_pred = np.squeeze(breast_unet.predict(np.expand_dims(image_gray, axis=0), verbose=0), axis=0)
    return mask_pred  # Return the raw prediction probabilities for the heatmap

def predict_breast_tumor(image):
    image_resized = cv2.resize(image, (128, 128))
    prediction = breast_tumor_classifier.predict(image_resized.reshape(-1, 128, 128, 3), verbose=0)
    return int(np.argmax(prediction))  # Assuming 0: Normal, 1: Benign, 2: Malignant


def overlay_heatmap(image, mask):
    mask_float = mask.astype(float)
    resized_mask = cv2.resize(mask_float, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_mask), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    return overlayed_image

PIXELS_TO_MM = 1  # Example conversion factor

def find_mask_bbox(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        w_mm = w * PIXELS_TO_MM
        h_mm = h * PIXELS_TO_MM
        return x, y, w_mm, h_mm
    return None, None, None, None

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if svm_predict(image) == -1:
        return jsonify({"error": "Uploaded image is not a recognized scan type"}), 400


    scan_type = predict_scan_type(image)
    scan_type_str = 'Brain Scan' if scan_type == 0 else 'Breast Scan'
    tumor_result = predict_tumor(image)

    if scan_type == 0:
        tumor_result_str = 'No Tumor' if tumor_result == 0 else 'Tumor'
        response_data = {
            'scan_type': scan_type_str,
            'tumor_result': tumor_result_str
        }

        if tumor_result == 1:
            mask_pred = predict_mask(image)
            x, y, w_mm, h_mm = find_mask_bbox((mask_pred > 0.5).astype(np.uint8))
            response_data.update({
                'x': x, 'y': y, 'width_mm': w_mm, 'height_mm': h_mm
            })

            overlayed_image = overlay_heatmap(image, mask_pred)
            original_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
            overlaid_image = cv2.imencode('.jpg', overlayed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
            response_data.update({
                'original_image': base64.b64encode(original_image).decode('utf-8'),
                'overlaid_image': base64.b64encode(overlaid_image).decode('utf-8')
            })
            return jsonify(response_data)
        elif tumor_result == 0:
            original_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
            response_data.update({
                'original_image': base64.b64encode(original_image).decode('utf-8'),
            })
            return jsonify(response_data)
    elif scan_type == 1:
        tumor_category = predict_breast_tumor(image)
        category_map = {0: 'Normal', 1: 'Malignant', 2: 'Benign'}
        tumor_result_str = category_map[tumor_category]

        response_data = {
            'scan_type': scan_type_str,
            'tumor_result': tumor_result_str
        }

        if tumor_category in [1, 2]:  # Benign or Malignant
            mask_pred = predict_breast_mask(image)
            x, y, w_mm, h_mm = find_mask_bbox((mask_pred > 0.5).astype(np.uint8))
            response_data.update({
                'x': x, 'y': y, 'width_mm': w_mm, 'height_mm': h_mm
            })

            overlayed_image = overlay_heatmap(image, mask_pred)
            overlaid_image = cv2.imencode('.jpg', overlayed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
            original_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()

            response_data.update({
                'original_image': base64.b64encode(original_image).decode('utf-8'),
                'overlaid_image': base64.b64encode(overlaid_image).decode('utf-8'),
                'tumor_result': tumor_result_str + ' Tumor'
            })
            return jsonify(response_data)
        elif tumor_category == 0:
            original_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])[1].tobytes()
            response_data.update({
                'original_image': base64.b64encode(original_image).decode('utf-8'),
                'tumor_result': 'No Tumor'
            })
            return jsonify(response_data)

if __name__ == '__main__':
    app.run( host='0.0.0.0', )

