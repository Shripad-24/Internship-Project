import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

# --- BASIC CONFIG ---
dataset_path = 'C:/Users/joshi/Internship Project/dataset/train'
img_size = 224
cld_size = 8
cld_feat_len = 108  # 3 * 6 * 6
X_img = []
X_cld = []
y = []
class_names = sorted(os.listdir(dataset_path))

# --- READ IMAGES AND EXTRACT FEATURES ---
for idx, classname in enumerate(class_names):
    class_dir = os.path.join(dataset_path, classname)
    for fname in os.listdir(class_dir):
        img_path = os.path.join(class_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        X_img.append(img)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        features = []
        for i in range(3):
            chan = cv2.resize(ycrcb[:,:,i], (cld_size, cld_size)).astype(np.float32)
            dct_chan = cv2.dct(chan)
            features.extend(dct_chan[:6, :6].flatten())
        X_cld.append(np.array(features))
        y.append(idx)

X_img = np.array(X_img) / 255.0
X_cld = np.array(X_cld)
y = np.array(y)

# --- TRAIN/TEST SPLIT ---
X_img_train, X_img_test, X_cld_train, X_cld_test, y_train, y_test = train_test_split(
    X_img, X_cld, y, test_size=0.2, random_state=42, stratify=y
)

# --- MODEL BUILDING ---
img_input = Input(shape=(img_size, img_size, 3))
cld_input = Input(shape=(cld_feat_len,))
x = Conv2D(32, (3,3), activation='relu')(img_input)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
cld_dense = Dense(64, activation='relu')(cld_input)
merged = Concatenate()([x, cld_dense])
merged = Dense(128, activation='relu')(merged)
merged = Dropout(0.5)(merged)
output = Dense(len(class_names), activation='softmax')(merged)

model = Model([img_input, cld_input], output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- TRAINING ---
history = model.fit(
    [X_img_train, X_cld_train], y_train,
    validation_split=0.15,
    epochs=50,
    batch_size=32
)

# --- EVALUATION ---
test_loss, test_acc = model.evaluate([X_img_test, X_cld_test], y_test)
print("Test accuracy:", test_acc)


# --- SAVE MODEL ---
model.save('CLD_CNN_Model.h5')
print("Model has been saved as 'CLD_CNN_Model'")
