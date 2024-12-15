import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from keras.models import load_model
import torch
from PIL import Image, ImageDraw
import numpy as np
import gdown
import cv2

app = Flask(__name__)
CORS(app)

# Class Mapping for Classification
CLASS_NAMES = {
    "Class 0": "Healthy Eye",
    "Class 1": "Cataract Detected"
}

# Download models if they don't exist
def download_models():
    if not os.path.exists("models"):
        os.makedirs("models")

    # Classification Model
    if not os.path.exists("models/CNN_No_Dropout.h5"):
        print("Downloading Classification Model...")
        gdown.download(
            "https://drive.google.com/uc?id=1vXLJXeoNySNE2ltN7UMD1aAnNu2zgoyb",
            "models/CNN_No_Dropout.h5",
            quiet=False
        )

    # Segmentation Model
    if not os.path.exists("models/best.pt"):
        print("Downloading Segmentation Model...")
        gdown.download(
            "https://drive.google.com/uc?id=18p5Re6-d1mdc17LskkUKru-ZtnNVundM",
            "models/best.pt",
            quiet=False
        )

    # Detection Model
    if not os.path.exists("models/best.torchscript"):
        print("Downloading Detection Model...")
        gdown.download(
            "https://drive.google.com/uc?id=1yy1hLydU6-s45b38GdaceqCVI34GH22W",
            "models/best.torchscript",
            quiet=False
        )

# Download models when the app starts
download_models()

# Load the models
try:
    print("Loading classification model...")
    classification_model = load_model("models/CNN_No_Dropout.h5")
    print("Classification model loaded successfully.")
except Exception as e:
    print(f"Failed to load classification model: {e}")

try:
    print("Loading segmentation model...")
    segmentation_model = torch.load("models/best.pt", map_location=torch.device("cpu"))
    print("Segmentation model loaded successfully.")
except Exception as e:
    print(f"Failed to load segmentation model: {e}")

try:
    print("Loading detection model...")
    detection_model = torch.jit.load("models/best.torchscript", map_location=torch.device("cpu"))
    print("Detection model loaded successfully.")
except Exception as e:
    print(f"Failed to load detection model: {e}")

# Helper function to preprocess images for classification
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Error occurred: {e}")
    return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/")
def home():
    return "Backend is running!"

# Classification Endpoint
@app.route("/classify", methods=["POST"])
def classify():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_path = "uploaded_image.jpg"
        file.save(file_path)

        # Preprocess and classify
        image = preprocess_image(file_path)
        prediction = classification_model.predict(image)
        os.remove(file_path)  # Clean up the uploaded file

        # Determine the class label
        predicted_class = "Class 1" if prediction[0][0] > 0.5 else "Class 0"
        class_name = CLASS_NAMES.get(predicted_class, "Unknown")
        print(f"Classification Result: {class_name}")

        return jsonify({"classification_result": class_name})
    except Exception as e:
        print(f"Error in /classify endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Segmentation Endpoint
@app.route("/segment", methods=["POST"])
def segment():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_path = "uploaded_image.jpg"
        file.save(file_path)

        # Perform segmentation
        image = Image.open(file_path).convert("RGB")
        image_tensor = torch.unsqueeze(torch.tensor(np.array(image)).permute(2, 0, 1), 0)
        result = segmentation_model(image_tensor)

        # Generate a binary mask
        mask = result[0].detach().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask).convert("L").resize(image.size)
        segmented_image = Image.composite(image, Image.new("RGB", image.size, (255, 0, 0)), mask_image)

        # Save and return the segmented image
        segmented_image_path = "segmented_image.jpg"
        segmented_image.save(segmented_image_path)
        os.remove(file_path)
        return send_file(segmented_image_path, mimetype="image/jpeg")
    except Exception as e:
        print(f"Error in /segment endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# Detection Endpoint
@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_path = "uploaded_image.jpg"
        file.save(file_path)

        # Perform detection
        image = cv2.imread(file_path)
        image_tensor = torch.unsqueeze(torch.tensor(image).permute(2, 0, 1), 0)
        result = detection_model(image_tensor)

        # Annotate image with bounding boxes
        detection_output = result.tolist()
        for detection in detection_output[0]:
            confidence = detection[0]
            x1, y1, x2, y2 = map(int, detection[1:5])
            if confidence > 0.5:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save and return the detected image
        detected_image_path = "detected_image.jpg"
        cv2.imwrite(detected_image_path, image)
        os.remove(file_path)
        return send_file(detected_image_path, mimetype="image/jpeg")
    except Exception as e:
        print(f"Error in /detect endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
