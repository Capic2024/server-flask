import cv2
import dlib
import numpy as np

# dlib 얼굴 감지기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 얼굴의 방향을 판별하는 함수
def determine_if_profile_face(shape):
    # 눈과 코의 좌표를 가져옵니다.
    left_eye_x = shape.part(36).x
    left_eye_y = shape.part(36).y
    right_eye_x = shape.part(45).x
    right_eye_y = shape.part(45).y
    nose_x = shape.part(30).x
    nose_y = shape.part(30).y

    # 눈과 코의 위치를 비교하여 얼굴의 방향을 판별합니다.
    # 눈의 중심과 코의 위치를 이용하여 각도를 계산합니다.
    # 이 예시에서는 단순히 각도를 이용하여 판별합니다.
    # 예를 들어, 특정 각도 이상이면 옆모습 얼굴로 판별할 수 있습니다.
    angle_threshold = 20  # 적절한 각도 임계값 설정 (예: 20도)

    # 눈과 코의 위치를 이용하여 각도를 계산합니다.
    angle = calculate_angle(left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y)

    # 각도가 임계값보다 크면 옆모습 얼굴로 판별합니다.
    if angle > angle_threshold:
        return True
    else:
        return False

def calculate_angle(x1, y1, x2, y2, x3, y3):
    # 두 벡터의 내적을 이용하여 각도를 계산합니다.
    # 내적을 이용한 각도 계산: https://en.wikipedia.org/wiki/Dot_product#Geometric_definition
    v1 = np.array([x1 - x3, y1 - y3])  # (x1, y1)에서 (x3, y3)를 향하는 벡터
    v2 = np.array([x2 - x3, y2 - y3])  # (x2, y2)에서 (x3, y3)를 향하는 벡터
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(cosine_angle) * (180 / np.pi)  # 라디안을 도로 변환
    return angle


# 모자이크 처리할 얼굴의 특징점 좌표
target_face_landmarks = None

# 모자이크 처리할 얼굴 이미지 로드
target_face_img = cv2.imread("goognyoo.png")
target_face_gray = cv2.cvtColor(target_face_img, cv2.COLOR_BGR2GRAY)
target_face_rects = detector(target_face_gray, 0)
if target_face_rects:
    target_face_landmarks = predictor(target_face_gray, target_face_rects[0])

# 동영상 불러오기
video_capture = cv2.VideoCapture('video.mp4')

while True:
    # 프레임 단위로 동영상 읽기
    ret, frame = video_capture.read()
    if not ret:
        break

    # 감지된 얼굴 모자이크 처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)

        # 얼굴의 방향 판별
        is_profile_face = determine_if_profile_face(shape)

        # 옆모습 얼굴이 아닌 경우에만 모자이크 처리
        if not is_profile_face:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            face = frame[y:y+h, x:x+w]
            face = cv2.GaussianBlur(face, (99, 99), 30)  # 모자이크 처리
            frame[y:y+h, x:x+w] = face

    # 화면에 출력
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
