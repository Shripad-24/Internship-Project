import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
import os

# Paths
train_dir = 'C:/Users/joshi/Internship Project/resnet50_dataset/train'
val_dir = 'C:/Users/joshi/Internship Project/resnet50_dataset/train'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
res_labels = np.load("res_labels.npy")

# 1. Improved data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 2. Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# class_labels = list(train_generator.class_indices.keys())
# with open("class_labels.pkl", "wb") as f:
#     pickle.dump(class_labels, f)

# 3. Load ResNet50 without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Initially freeze base

# 4. Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(len(res_labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# 5. Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2, min_lr=1e-6),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# 7. Train (initial phase)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# 8. Fine-tune the top ResNet50 layers
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Freeze bottom layers
    layer.trainable = False

# Recompile with lower LR for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 9. Fine-tune
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# 10. Save final model
model.save("resnet50_image_classifier_improved.h5")

# 11. Plot accuracy and loss
def plot_history(h1, h2=None):
    acc = h1.history['accuracy'] + (h2.history['accuracy'] if h2 else [])
    val_acc = h1.history['val_accuracy'] + (h2.history['val_accuracy'] if h2 else [])
    loss = h1.history['loss'] + (h2.history['loss'] if h2 else [])
    val_loss = h1.history['val_loss'] + (h2.history['val_loss'] if h2 else [])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.show()

plot_history(history, fine_tune_history)
