import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# Load trained model
model = load_model("disease_cnn_model.h5")
train_dir='C:/Users/joshi/Internship Project/EHD_Train/Train_EHD'
# Load correct class index mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Invert class_indices to get label names from indices
index_to_class = {v: k for k, v in class_indices.items()}

# Image settings
img_height, img_width = 256, 256

# Function to predict a single image
def predict_user_image(img_path):
    try:
        img = load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        print(f"Image array shape before expand_dims: {img_array.shape}")  # Debug print
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Image array shape after expand_dims: {img_array.shape}")  # Debug print

        predictions = model.predict(img_array)
        print(f"Raw model predictions: {predictions}")  # Debug print
        predicted_index = np.argmax(predictions)
        predicted_label = index_to_class[predicted_index]

        print(f"\nPredicted Disease: {predicted_label}")

    except Exception as e:
        print(f"Error: {e}")

# Evaluate model accuracy
def evaluate_model_accuracy():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    val_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )

    _, acc = model.evaluate(val_generator, verbose=0)
    print(f"\nTotal Model Accuracy on Validation Set: {acc * 100:.2f}%")

# Run prediction
img_path = input("Enter full path to an EHD image:\n").strip()
predict_user_image(img_path)

# Print model accuracy
evaluate_model_accuracy()
