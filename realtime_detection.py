import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load trained model and scaler
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

IMG_SIZE = 64

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI for hand
    roi = frame[100:400, 350:650]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    equalized = cv2.equalizeHist(gray)

    # Resize
    resized = cv2.resize(equalized, (IMG_SIZE, IMG_SIZE))

    # HOG feature extraction (SAME AS TRAINING)
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # Scale features
    features = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = knn.predict(features)[0]

    # Draw UI
    cv2.rectangle(frame, (350, 100), (650, 400), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Frame", frame)
    cv2.imshow("Processed ROI", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
