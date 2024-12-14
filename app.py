from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from file_download import download_models
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the models
classification_model = load_model("models/CNN_No_Dropout.h5")
segmentation_model = load_model("models/Segmentation_Model.h5")
detection_model = load_model("models/Detection_Model.h5")

# Helper function to preprocess the image for classification
def preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path).convert("RGB").resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Helper function to preprocess the image for segmentation and detection
def preprocess_for_segmentation(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert("RGB").resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    # Process the image for classification
    image_for_classification = preprocess_image(file_path)
    classification_result = (classification_model.predict(image_for_classification) > 0.5).astype("int32")[0][0]

    # Process the image for segmentation
    image_for_segmentation = preprocess_for_segmentation(file_path)
    segmentation_result = segmentation_model.predict(image_for_segmentation)[0]

    # Save segmentation result as an image
    segmentation_output_path = "static/segmentation_output.jpg"
    segmentation_image = (segmentation_result * 255).astype(np.uint8)
    segmentation_image = Image.fromarray(segmentation_image)
    segmentation_image.save(segmentation_output_path)

    # Process the image for object detection
    detection_output_path = "static/detection_output.jpg"
    detection_image = Image.open(file_path)
    detection_image.save(detection_output_path)

    os.remove(file_path)
    return jsonify({
        'classification': int(classification_result),
        'segmentation': segmentation_output_path,
        'detection': detection_output_path
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
