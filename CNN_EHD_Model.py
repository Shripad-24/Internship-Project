import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parameters
image_size = (224, 224)
num_classes = 7
dataset_path = 'C:/Users/joshi/Internship Project/dataset/train'

# Function to extract EHD (Edge Histogram Descriptor)
def extract_ehd_features(image, bins=8):
    edges = cv2.Canny(image, 100, 200)
    hist = cv2.calcHist([edges], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Load Images + Labels + EHD
X_img = []
X_ehd = []
y = []

class_names = sorted(os.listdir(dataset_path))
class_indices = {cls: idx for idx, cls in enumerate(class_names)}

for label in class_names:
    folder = os.path.join(dataset_path, label)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ehd = extract_ehd_features(gray)

        X_img.append(img / 255.0)  # Normalize image
        X_ehd.append(ehd)
        y.append(class_indices[label])

X_img = np.array(X_img)
X_ehd = np.array(X_ehd)
y = to_categorical(y, num_classes=num_classes)

# Train-validation split
X_img_train, X_img_val, X_ehd_train, X_ehd_val, y_train, y_val = train_test_split(
    X_img, X_ehd, y, test_size=0.2, random_state=42
)

# CNN Branch
image_input = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)

# EHD Branch
ehd_input = Input(shape=(X_ehd.shape[1],))
y = Dense(64, activation='relu')(ehd_input)
y = Dropout(0.3)(y)

# Concatenate CNN + EHD
combined = concatenate([x, y])
z = Dense(256, activation='relu')(combined)
z = Dropout(0.4)(z)
z = Dense(num_classes, activation='softmax')(z)

# Build and Compile
model = Model(inputs=[image_input, ehd_input], outputs=z)
model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    ModelCheckpoint('ehd_cnn_model.h5', save_best_only=True)
]

# Train Model
history = model.fit(
    [X_img_train, X_ehd_train], y_train,
    validation_data=([X_img_val, X_ehd_val], y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Final Evaluation
val_loss, val_acc = model.evaluate([X_img_val, X_ehd_val], y_val)
print(f"\nFinal Validation Accuracy (EHD+CNN): {val_acc * 100:.2f}%")

# Save Model
model.save("ehd_cnn_model.h5")
