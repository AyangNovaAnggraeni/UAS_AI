import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/bisindo_model.h5")

# Load label names if available
label_path = "model/labels.npy"
if os.path.exists(label_path):
    classes = np.load(label_path)
else:
    # Default to A-Z
    classes = np.array([chr(i) for i in range(65, 91)])  # ['A', 'B', ..., 'Z']

# Load and preprocess test image
img_path = "images/train/Z/augmented_image_1.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from '{img_path}'")
    exit()

# Resize and normalize
img = cv2.resize(img, (64, 64))
img = img.astype('float32') / 255.0

# Handle grayscale or color input
if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
elif img.shape[-1] == 1:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Reshape to model input
img = np.expand_dims(img, axis=0)  # (1, 64, 64, 3)

# Make prediction
prediction = model.predict(img, verbose=0)
predicted_index = np.argmax(prediction)
confidence = np.max(prediction)

# Apply threshold
THRESHOLD = 0.70
if confidence >= THRESHOLD:
    predicted_label = classes[predicted_index]
    print(f"Prediction: {predicted_label} ({confidence*100:.2f}%)")
else:
    print(f"Low confidence ({confidence*100:.2f}%), prediction uncertain.")

