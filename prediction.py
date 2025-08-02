import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load CLD and EHD labels
cld_labels = np.load("cld_labels.npy")
ehd_labels = np.load("ehd_labels.npy")
res_labels = np.load("res_labels.npy")
vgg_labels = np.load("vgg_labels.npy")

# Load models
cnn_model = load_model("cnn_model.h5")
resnet_model = load_model("resnet50_image_classifier_improved.h5")
cld_model = load_model("cld_cnn_model.h5")
ehd_model = load_model("ehd_cnn_model.h5")
vgg_model = load_model("vgg16_hibiscus_model.h5")

# Parameters
img_size = 224


def extract_cld(img):
    cld_features = []
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    for i in range(3):  
        chan = cv2.resize(ycrcb[:,:,i], (8, 8)).astype(np.float32)
        dct_chan = cv2.dct(chan)
        cld_features.extend(dct_chan[:6, :6].flatten())
    return np.array(cld_features)


def extract_ehd(img, bins=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    hist = cv2.calcHist([edges], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


# 1. Load input image
def load_input_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_size, img_size))
    return img

# 2. Predict using all models
def predict_all_models(image_path):
    img = load_input_image(image_path)
    input_img = img.astype('float32') / 255.0
    input_img_exp = np.expand_dims(input_img, axis=0)

    # CNN Prediction
    cnn_pred = cnn_model.predict(input_img_exp)
    cnn_class = np.argmax(cnn_pred)
    cnn_label = list(cld_labels)[cnn_class]
    cnn_conf = np.max(cnn_pred) * 100

    # ResNet Prediction
    resnet_input = preprocess_input(np.expand_dims(img.astype('float32'), axis=0))
    resnet_pred = resnet_model.predict(resnet_input)
    resnet_class = np.argmax(resnet_pred)
    resnet_label = list(res_labels)[resnet_class]
    resnet_conf = np.max(resnet_pred) * 100

    # CLD+CNN Prediction
    cld_feat = extract_cld(img).reshape(1, -1)  
    cld_pred = cld_model.predict([input_img_exp,cld_feat])
    cld_class = np.argmax(cld_pred)
    cld_label = cld_labels[cld_class]
    cld_conf = np.max(cld_pred) * 100

    # EHD+CNN Prediction
    ehd_feat = extract_ehd(img).reshape(1, -1)
    ehd_pred = ehd_model.predict([input_img_exp,ehd_feat])
    ehd_class = np.argmax(ehd_pred)
    ehd_label = ehd_labels[ehd_class]
    ehd_conf = np.max(ehd_pred) * 100

    #VGG16 Prediction
    # Preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  
    vgg_pred = vgg_model.predict(img)
    vgg_label = vgg_labels[np.argmax(vgg_pred)]
    vgg_conf = np.max(vgg_pred) * 100

    # Final Output
    print("\nPrediction Results:")
    print(f"CNN Prediction: {cnn_label} ({cnn_conf:.2f}%)")
    print(f"ResNet-50 Prediction: {resnet_label} ({resnet_conf:.2f}%)")
    print(f"CLD + CNN Prediction: {cld_label} ({cld_conf:.2f}%)")
    print(f"EHD + CNN Prediction: {ehd_label} ({ehd_conf:.2f}%)")
    print(f"VGG16 Prediction: {vgg_label} ({vgg_conf:.2f}%)")

# Example usage
if __name__ == "__main__":
    image_path = input("Enter path to test image: ")
    predict_all_models(image_path)
