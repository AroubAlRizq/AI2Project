import os
import gdown

def download_models():
    if not os.path.exists("models"):
        os.makedirs("models")

    # Download CNN_No_Dropout.h5
    if not os.path.exists("models/CNN_No_Dropout.h5"):
        print("Downloading CNN_No_Dropout.h5...")
        gdown.download("https://drive.google.com/uc?id=1vXLJXeoNySNE2ltN7UMD1aAnNu2zgoyb", "models/CNN_No_Dropout.h5", quiet=False)

    # Download CNN_With_Dropout.h5
    if not os.path.exists("models/CNN_With_Dropout.h5"):
        print("Downloading CNN_With_Dropout.h5...")
        gdown.download("https://drive.google.com/uc?id=1ye0WGRrOYGkZbbMHna-4FW0yYqwVxlT1", "models/CNN_With_Dropout.h5", quiet=False)

# Call the function when the app starts
download_models()
