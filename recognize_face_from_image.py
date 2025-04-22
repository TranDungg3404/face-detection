import face_recognition
import cv2
import os
import numpy as np

# === Load khuôn mặt đã biết ===
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(person_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Chuyển ảnh sang RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

# === Nhận diện ảnh test ===
test_image_path = "test1.jpg"
image = cv2.imread(test_image_path)
if image is None:
    print("Không thể mở ảnh test.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(image_rgb)
face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

# Hiển thị kết quả
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(distances)
    name = "Unknown"
    accuracy = 0.0

    if distances[best_match_index] < 0.45:  
        name = known_names[best_match_index]
        accuracy = 1 - distances[best_match_index]

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    label = f"{name} ({accuracy * 100:.2f}%)"
    cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

# Hiển thị ảnh đúng kích thước gốc
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", image.shape[1], image.shape[0])
cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
