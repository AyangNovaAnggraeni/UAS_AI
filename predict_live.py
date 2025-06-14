import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

# Load the trained model (adjust path as needed)
model = tf.keras.models.load_model('model/bisindo_model.h5')

# Define threshold for accepting predictions
CONFIDENCE_THRESHOLD = 0.7
LETTER_COOLDOWN = 1  # seconds between adding same letter
prediction_buffer = []
BUFFER_SIZE = 5


# Initialize
buffered_letters = []
current_word = ''
last_letter = ''
last_letter_time = datetime.now()
sentence = ''
last_word_time = datetime.now()
WORD_COMMIT_DELAY = 2.0 

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_frame(frame, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))

    # Check if roi has 3 channels
    if len(roi.shape) == 2:  # (64,64) -> grayscale
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    roi = roi.reshape(1, 64, 64, 3) / 255.0
    return roi


def get_roi_coords(frame_shape):
    h, w = frame_shape[:2]
    # Dynamically calculate ROI (center square)
    size = int(min(h, w) * 0.5)
    x1 = w//2 - size//2
    y1 = h//2 - size//2
    x2 = x1 + size
    y2 = y1 + size
    return x1, y1, x2, y2

def clear_sentence():
    global current_word, buffered_letters
    current_word = ''
    buffered_letters.clear()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural view
    frame = cv2.flip(frame, 1)

    # Get dynamic ROI
    roi_coords = get_roi_coords(frame.shape)
    x1, y1, x2, y2 = roi_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Put hand here", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Preprocess
    roi_input = preprocess_frame(frame, roi_coords)

    # Predict 
    predictions = model.predict(roi_input, verbose=0)
    predicted_class = chr(np.argmax(predictions) + 65)
    confidence = np.max(predictions)

    # Add to buffer
    prediction_buffer.append(predicted_class)
    if len(prediction_buffer) > BUFFER_SIZE:
        prediction_buffer.pop(0)

    # Voting
    most_common = max(set(prediction_buffer), key=prediction_buffer.count)


    # Show prediction
    cv2.putText(frame, f'Prediction: {predicted_class} ({confidence*100:.1f}%)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Accept only high-confidence predictions
    current_time = datetime.now()
    # Tambahkan huruf jika confident dan cukup waktu
    if confidence > CONFIDENCE_THRESHOLD and most_common.isalpha():
        if most_common != last_letter:
            if (current_time - last_letter_time).total_seconds() > LETTER_COOLDOWN:
                buffered_letters.append(most_common)
                last_letter = most_common
                last_letter_time = current_time
                last_word_time = current_time

        current_word = ''.join(buffered_letters)

    if buffered_letters and (datetime.now() - last_word_time).total_seconds() > WORD_COMMIT_DELAY:
        sentence += ''.join(buffered_letters) + ' '
        buffered_letters.clear()
        current_word = ''

    cv2.putText(frame, f'Sentence: {sentence}',
            (10, frame.shape[0]-70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)

    # Show current word
    cv2.putText(frame, f'Current Word: {current_word}',
                (10, frame.shape[0]-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Clear sentence button (use 'c' key)
    cv2.putText(frame, f'Press "c" to Clear',
                (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Sign Language to Text', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('c'):
        clear_sentence()

cap.release()
cv2.destroyAllWindows()


