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
        gdown.download("https://drive.google.com/uc?id=18p5Re6-d1mdc17LskkUKru-ZtnNVundM", "models/best.pt", quiet=False)

    # Download Detection model (.torchscript for PyTorch)
    if not os.path.exists("models/best.torchscript"):
        print("Downloading Detection_Model.torchscript...")
        gdown.download("https://drive.google.com/uc?id=1yy1hLydU6-s45b38GdaceqCVI34GH22W", "models/best.torchscript", quiet=False)

if __name__ == "__main__":
    download_models()
