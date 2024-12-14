import os
import gdown

def download_models():
    if not os.path.exists("models"):
        os.makedirs("models")

    # Download CNN model
    if not os.path.exists("models/CNN_Model.h5"):
        print("Downloading CNN_Model.h5...")
        gdown.download("https://drive.google.com/uc?id=1vXLJXeoNySNE2ltN7UMD1aAnNu2zgoyb", "models/CNN_Model.h5", quiet=False)

    # Download Segmentation model
    if not os.path.exists("models/Segmentation_Model.h5"):
        print("Downloading Segmentation_Model.h5...")
        gdown.download("https://drive.google.com/uc?id=YOUR_SEGMENTATION_MODEL_ID", "models/Segmentation_Model.h5", quiet=False)

    # Download Detection model
    if not os.path.exists("models/Detection_Model.h5"):
        print("Downloading Detection_Model.h5...")
        gdown.download("https://drive.google.com/uc?id=YOUR_DETECTION_MODEL_ID", "models/Detection_Model.h5", quiet=False)

# Call the function when the app starts
if __name__ == "__main__":
    download_models()
