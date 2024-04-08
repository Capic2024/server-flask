import cv2
import face_recognition
import os

video_path = 'video.mp4'
output_folder = 'video_face'  # 얼굴 이미지 저장할 폴더
known_encodings = []  # 기존의 "얼굴 인코딩"
known_names = []  # 해당 사람의 이름

# 출력할 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 비디오를 읽기 위한 VideoCapture 객체 생성
video_capture = cv2.VideoCapture(video_path)

# 프레임을 저장할 배열 초기화
frames = []

# 비디오의 각 프레임을 순회하여 배열에 저장
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break
    frames.append(frame)

# 자원 해제
video_capture.release()

# 배열에 저장된 각 프레임에 대해 처리
for frame_index, frame in enumerate(frames):
    if frame is None:
        continue

    face_locations = face_recognition.face_locations(frame)
    #face_locations = face_recognition.face_locations(frame, model="cnn") # face-recornition의 CNN 모델 선택
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_index, (top, right, bottom, left), face_encoding in zip(range(len(face_locations)), face_locations,
                                                                      face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        else:
            name = f"person_{len(known_encodings) + 1}"
            known_encodings.append(face_encoding)
            known_names.append(name)
            person_folder = os.path.join(output_folder, name)
            if not os.path.exists(person_folder):
                os.makedirs(person_folder)

        face_image = frame[top:bottom, left:right]
        output_path = os.path.join(person_folder, f"face_{frame_index}_{face_index}.jpg")
        cv2.imwrite(output_path, face_image)

print(f"모든 얼굴 이미지가 {output_folder} 폴더에 인물별로 저장되었습니다.")
