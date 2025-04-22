import cv2
import face_recognition
import os

# === TẢI DỮ LIỆU HUẤN LUYỆN TỪ sample_images ===
print("[INFO] Đang tải dữ liệu khuôn mặt mẫu...")
dataset_path = "known_faces"
known_encodings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Không thể đọc ảnh: {img_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"[WARN] Không tìm thấy khuôn mặt trong ảnh: {img_path}")

print(f"[INFO] Đã nạp {len(known_encodings)} khuôn mặt mẫu.")

# === NHẬN DIỆN QUA WEBCAM/VIDEO ===
video_capture = cv2.VideoCapture(0)  # hoặc truyền vào đường dẫn video

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Không đọc được khung hình.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matched_idxs:
                matched_name = known_names[i]
                counts[matched_name] = counts.get(matched_name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    # Vẽ kết quả lên frame
    for ((top, right, bottom, left), name) in zip(face_locations, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 10 if top - 10 > 10 else top + 20
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Face Recognition (Press Q to Quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
