import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and preprocess image
image = cv2.imread("C:/Users/joshi/Downloads/download_(8).jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (256, 256))  # Resize for consistent processing

# Step 2: Define EHD edge filters
filters = {
    "Vertical": np.array([[-1,  0, 1], [-2,  0, 2], [-1,  0, 1]]),
    "Horizontal": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "45_Diagonal": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
    "135_Diagonal": np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
    "Non_Directional": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
}

# Step 3: Apply each filter and store results
filtered_images = {}
for name, kernel in filters.items():
    filtered = cv2.filter2D(gray, -1, kernel)
    filtered_images[name] = cv2.convertScaleAbs(filtered)  # take abs & convert to uint8

# Step 4: Combine all edge responses for visualization (strongest only)
combined = np.zeros_like(gray)
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        responses = [filtered_images[key][i, j] for key in filters]
        combined[i, j] = max(responses)  # strongest edge response

# Step 5: Threshold to get clear binary edges
_, binary_edges = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)

# Step 6: Show results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale Image")

plt.subplot(1, 2, 2)
plt.imshow(binary_edges, cmap='gray')
plt.title("Plant Edges via EHD")

plt.tight_layout()
plt.show()
