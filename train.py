import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

X = []
y = []

DATASET_PATH = "dataset"
IMG_SIZE = 64

for label in range(6):   # 0–5 fingers
    folder = os.path.join(DATASET_PATH, str(label))

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Histogram Equalization
        img = cv2.equalizeHist(img)

        # HOG feature extraction
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

# 🔹 Feature Scaling (VERY IMPORTANT FOR KNN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 Train-Test Split (sanity check)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Train KNN
knn = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    metric='euclidean'
)
knn.fit(X_train, y_train)

# 🔹 Test Accuracy
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# 🔹 Save model & scaler
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ KNN model & scaler saved successfully")
