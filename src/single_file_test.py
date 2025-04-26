# test_single_image.py

import torch
from PIL import Image

from data_handler import Preproccessor
from model import CNN

model = CNN()

# === Configuration ===
MODEL_PATH = "./base_model/trial-1_calibrated_80_20_split_96_acc.pt"        # <-- Path to your trained model
#MODEL_PATH = "./base_model/base_model_922_acc.pt" #"./base_model/trial-1_80_20_split_96_acc.pt"        # <-- Path to your trained model
IMAGE_PATH = "./datasets/diego_1.jpg"     # <-- Path to the single image you want to test

# === Image Preprocessing ===
img = Image.open(IMAGE_PATH).convert("RGB")
preprocess = Preproccessor()
img_tensor = preprocess.process( img )
img_tensor = img_tensor.unsqueeze( 0 )

# === Prediction ===
model.load_model( MODEL_PATH )

prediction, probability = model.test_image( img_tensor )

print(f"Predicted Class: {prediction} (1 = Gingivitis, 0 = Healthy)")
print(f"Probability Score: {float( probability.item() ):.4f}")

# === Done ===
