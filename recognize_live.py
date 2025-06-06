import cv2
import tensorflow as tf
import numpy as np
import os

img_height, img_width = 160, 160

# Load OpenCV's pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load your trained face recognition model
model = tf.keras.models.load_model("face_rec_model.h5")  # <-- Update with your model path

# Example: class names (update with your actual class names)
# Automatically get class names from your dataset directory
dataset_dir = "dataset"  
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

cap = cv2.VideoCapture(0)  # 0 is the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Extract the face ROI and preprocess for model input
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (img_width, img_height))
        img_array = tf.keras.utils.img_to_array(face_img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Pass img_array to your recognition model
        preds = model.predict(img_array)
        pred_label = class_names[np.argmax(preds)]

        # Display label above the rectangle
        cv2.putText(frame, pred_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Show the frame with rectangles and labels
    cv2.imshow('Live Face Capture', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()