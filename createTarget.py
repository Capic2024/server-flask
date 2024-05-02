import cv2
import face_recognition
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

def extract_and_identify_faces_from_video(video_path):
    face_encodings = []
    face_images = []
    identified_faces = []

    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        face_locations = face_recognition.face_locations(frame)
        current_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, current_encodings):
            face_image = frame[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            face_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            face_images.append(face_image_base64)
            face_encodings.append(encoding)

    for idx, encoding in enumerate(face_encodings):
        if not identified_faces:
            identified_faces.append([(face_images[idx], encoding.tolist())])
        else:
            matched = False
            for face_group in identified_faces:
                group_encodings = [enc for _, enc in face_group]
                avg_encoding = np.mean(group_encodings, axis=0)
                dist = np.linalg.norm(avg_encoding - encoding)
                if dist < 0.6:
                    face_group.append((face_images[idx], encoding.tolist()))
                    matched = True
                    break
            if not matched:
                identified_faces.append([(face_images[idx], encoding.tolist())])

    video_capture.release()
    return identified_faces


def save_faces(identified_faces):
    for i, face_group in enumerate(identified_faces):
        group_dir = f'faces/person_{i + 1}'
        os.makedirs(group_dir, exist_ok=True)

        for j, (face_base64, encoding) in enumerate(face_group):
            # Base64 문자열을 이미지로 디코드
            face_data = base64.b64decode(face_base64)
            face_array = np.array(Image.open(BytesIO(face_data)))  # PIL을 사용하여 이미지 객체를 생성하고, 이를 NumPy 배열로 변환

            # 이미지 저장
            cv2.imwrite(f'{group_dir}/face_{j}.jpg', cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR))  # OpenCV는 BGR 형식을 사용


