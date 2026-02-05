import cv2
import os

gesture_label = input("Enter gesture number (0-5): ")
save_path = f"dataset/{gesture_label}"

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # ROI (Region of Interest)
    roi = frame[100:400, 350:650]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    resized = cv2.resize(equalized, (64, 64))

    cv2.imshow("ROI", resized)
    cv2.rectangle(frame, (350,100), (650,400), (0,255,0), 2)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):  # press 'c' to capture
        img_path = f"{save_path}/{count}.jpg"
        cv2.imwrite(img_path, resized)
        count += 1
        print("Image saved:", img_path)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
