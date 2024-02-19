import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# MTCNN 초기화
mtcnn = MTCNN(keep_all=True)

# 모자이크할 얼굴을 식별할 이미지 로드
# image_path = os.path.join('//static', 'goognyoo.png')
face = cv2.imread('goognyoo.png')
face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

# 모자이크할 얼굴 좌표 식별
boxes, _ = mtcnn.detect(face_rgb)

# 얼굴 좌표가 있는 경우, 첫 번째 얼굴의 좌표를 저장
target_face_coords = boxes[0] if boxes is not None else None

# 동영상 불러오기
video_capture = cv2.VideoCapture('video.mp4')

def is_overlapping(box1, box2):
    """
    두 개의 박스(얼굴 영역)가 서로 겹치는지를 확인하는 함수
    :param box1: 첫 번째 박스의 좌표 (x1, y1, x2, y2)
    :param box2: 두 번째 박스의 좌표 (x1, y1, x2, y2)
    :return: 두 박스가 겹치면 True, 그렇지 않으면 False
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 두 박스가 서로 겹치는지를 확인하는 조건식
    overlapping = not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

    return overlapping

while True:
    # 프레임 단위로 동영상 읽기
    ret, frame = video_capture.read()
    if not ret:
        break

    # 감지된 얼굴 모자이크 처리
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            # 모자이크할 얼굴 좌표가 있고, 현재 얼굴이 해당 좌표와 겹치지 않는 경우 모자이크 처리
            if target_face_coords is not None and not is_overlapping(box, target_face_coords):
                # 얼굴 영역을 모자이크 처리
                x1, y1, x2, y2 = box.astype(int)
                face = frame[y1:y2, x1:x2]
                face = cv2.GaussianBlur(face, (99, 99), 30)  # 모자이크 처리
                frame[y1:y2, x1:x2] = face

    # 화면에 출력
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
