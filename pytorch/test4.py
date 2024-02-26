import cv2
import os

# 얼굴 감지기 초기화
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 추출된 프레임들이 담긴 디렉토리 경로
frames_dir = 'video_frames/'

# 특정 인물의 얼굴을 인식할 프레임의 디렉토리 경로
target_faces_dir = 'target_faces/'

# 특정 인물의 얼굴 이미지를 불러옵니다.
target_face = cv2.imread('goognyoo.png', cv2.IMREAD_GRAYSCALE)
target_face = cv2.imread('gong_u2.jpeg', cv2.IMREAD_GRAYSCALE)
target_face = cv2.imread('gong_u3.png', cv2.IMREAD_GRAYSCALE)
target_face = cv2.imread('gong_u4.jpg', cv2.IMREAD_GRAYSCALE)
target_face = cv2.imread('gong_u5.jpeg', cv2.IMREAD_GRAYSCALE)


# 추출된 프레임들을 하나씩 읽어옵니다.
for frame_file in os.listdir(frames_dir):
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)

    # 그레이 스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 감지된 얼굴들을 순회하면서 특정 인물의 얼굴과 비교
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # 추출된 얼굴과 특정 인물의 얼굴 비교
        # 이 부분은 특정 인물을 인식하는 방법에 따라 달라질 수 있습니다.
        # 예를 들어, 추출된 얼굴과 특정 인물의 얼굴 간의 유사도를 비교하여 일정 기준 이상인 경우 해당 얼굴을 특정 인물로 판단할 수 있습니다.
        # 여기서는 단순히 추출된 얼굴과 특정 인물의 얼굴이 일치하는지 여부를 확인하는 예제입니다.
        if roi_gray.shape == target_face.shape:
            difference = cv2.subtract(roi_gray, target_face)
            result = not np.any(difference)
            if result is True:
                print("Found target person's face in:", frame_file)
                # 인식된 얼굴 주변에 사각형 그리기
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 얼굴 인식 결과 표시
                cv2.putText(frame, 'Target Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 프레임 보여주기
    cv2.imshow('Frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
