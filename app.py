import os
import torch  # For loading PyTorch models
from ultralytics import YOLO  # For YOLO models
from tensorflow.keras.models import load_model  # For .h5 models
from file_download import download_models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# Force CPU usage for TensorFlow and PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Download models before loading
download_models()

# Load TensorFlow CNN model (.h5)
classification_model = load_model("models/CNN_Model.h5")

# Load PyTorch Segmentation model (YOLO format)
segmentation_model = YOLO("models/best.pt")  # Ensure this works with your segmentation model

# Load PyTorch Detection model (.torchscript)
detection_model = torch.jit.load("models/best.torchscript", map_location=torch.device("cpu"))
detection_model.eval()  # Set the model to evaluation mode


# Helper: Preprocess for CNN classification
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))  # Resize to model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 128, 128, 3)  # Add batch dimension
    return image_array


# Helper: Preprocess for PyTorch models
def preprocess_pytorch_image(image_path, size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size)  # Resize to model input size
    image_array = np.array(image).transpose(2, 0, 1) / 255.0  # Normalize and transpose to (C, H, W)
    image_tensor = torch.tensor(image_array).unsqueeze(0).float()  # Add batch dimension
    return image_tensor


@app.route("/")
def index():
    return "Backend is running!"


@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    # Preprocess and classify
    image = preprocess_image(file_path)
    prediction = classification_model.predict(image)
    os.remove(file_path)  # Clean up the uploaded file

    # Return the classification result
    result = "Class 1" if prediction[0][0] > 0.5 else "Class 0"
    return jsonify({"classification_result": result})


@app.route("/segment", methods=["POST"])
def segment():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    # Run segmentation using YOLO
    results = segmentation_model(file_path)
    segmentation_output_path = "static/segmentation_output.jpg"
    results[0].plot(save_dir="static", save_name="segmentation_output.jpg")

    os.remove(file_path)  # Clean up the uploaded file
    return jsonify({"segmentation_result": segmentation_output_path})


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    # Preprocess and detect
    image_tensor = preprocess_pytorch_image(file_path, size=(640, 640))
    with torch.no_grad():
        detection_output = detection_model(image_tensor)  # Forward pass

    os.remove(file_path)  # Clean up the uploaded file
    return jsonify({"detection_result": detection_output.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
