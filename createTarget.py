import base64

import cv2
import face_recognition
import os
import numpy as np
from flask import jsonify


def extract_and_identify_faces_from_video(video_path):
    face_encodings = []  # 얼굴별 인코딩 저장
    face_images = []  # 얼굴별 이미지 저장
    identified_faces = []  # 식별된 얼굴별 (객체) 저장

    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        face_locations = face_recognition.face_locations(frame)  # 현재 프레임에서 얼굴 위치 탐지
        current_encodings = face_recognition.face_encodings(frame, face_locations)  # 얼굴 위치에 대한 인코딩

        for (top, right, bottom, left), encoding in zip(face_locations, current_encodings):
            # 얼굴 이미지 추출
            face_image = frame[top:bottom, left:right]
            face_images.append(face_image)
            face_encodings.append(encoding)

    # 인식된 얼굴 분류
    for idx, encoding in enumerate(face_encodings):
        if not identified_faces:
            identified_faces.append([(face_images[idx], encoding)])
        else:
            matched = False
            for face_group in identified_faces:
                group_encodings = [enc for _, enc in face_group]
                avg_encoding = np.mean(group_encodings, axis=0)
                dist = np.linalg.norm(avg_encoding - encoding)
                if dist < 0.6:  # 같은 사람으로 판단하는 임계값
                    face_group.append((face_images[idx], encoding))
                    matched = True
                    break
            if not matched:
                identified_faces.append([(face_images[idx], encoding)])

    video_capture.release()
    print('end1')

    # 인식된 얼굴 이미지를 Base64로 인코딩하여 반환
    return save_faces(identified_faces)


def save_faces(identified_faces):
    face_base64_arrays = []  # Base64 인코딩된 이미지 배열을 저장할 리스트
    for group in identified_faces:
        group_base64_arrays = []
        for face, _ in group[:3]:  # 인물별 최대 3개까지
            _, buffer = cv2.imencode('.jpg', face)  # OpenCV를 사용하여 이미지를 바이트로 인코딩
            byte_array = buffer.tobytes()  # NumPy 배열을 바이트 배열로 변환
            base64_encoded = base64.b64encode(byte_array).decode('utf-8')  # Base64로 인코딩 후 문자열로 변환
            group_base64_arrays.append(base64_encoded)
        face_base64_arrays.append(group_base64_arrays)
    return face_base64_arrays  # 2차원 문자열 배열 반환




