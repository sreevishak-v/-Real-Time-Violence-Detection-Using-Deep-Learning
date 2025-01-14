import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
from datetime import datetime

# Load the trained model
model = load_model('C:/-Real-Time-Violence-Detection-Using-Deep-Learning/models12/violence_detection_model2.h5')

# Define the input image size for the model
IMG_SIZE = (128, 128)

# Define a function to preprocess each frame
def preprocess_frame(frame):
    # Convert BGR (OpenCV default) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image for resizing
    img = Image.fromarray(frame, 'RGB')
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Directory to save screenshots
screenshot_dir = "screenshots"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

# Start the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or provide a video file path

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting Real-Time Violence Detection... Press 'q' to quit.")

# Real-time video feed processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Make predictions
    prediction = model.predict(preprocessed_frame)[0][0]
    label = "Violence" if prediction > 0.7 else "Non-Violence"  # Adjust threshold (0.7) for confidence
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Display the result on the frame
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)  # Red for violence, Green for non-violence
    cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Take a screenshot if violence is detected
    if label == "Violence" and confidence > 0.7:  # Confidence threshold for saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshot_dir, f"violence_{timestamp}.png")
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved: {screenshot_path}")

    # Show the frame
    cv2.imshow("Real-Time Violence Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()