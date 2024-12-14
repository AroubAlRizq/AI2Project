import os
import gdown

def download_models():
    if not os.path.exists("models"):
        os.makedirs("models")

    # Download CNN model (TensorFlow .h5)
    if not os.path.exists("models/CNN_Model.h5"):
        print("Downloading CNN_Model.h5...")
        gdown.download("https://drive.google.com/uc?id=1vXLJXeoNySNE2ltN7UMD1aAnNu2zgoyb", "models/CNN_Model.h5", quiet=False)

    # Download Segmentation model (.pt for PyTorch)
    if not os.path.exists("models/best.pt"):
        print("Downloading Segmentation_Model.pt...")
        gdown.download("https://drive.google.com/uc?id=YOUR_SEGMENTATION_MODEL_ID", "models/Segmentation_Model.pt", quiet=False)

    # Download Detection model (.torchscript for PyTorch)
    if not os.path.exists("models/best.torchscript"):
        print("Downloading Detection_Model.torchscript...")
        gdown.download("https://drive.google.com/uc?id=YOUR_DETECTION_MODEL_ID", "models/Detection_Model.torchscript", quiet=False)

if __name__ == "__main__":
    download_models()
