import cv2

# 얼굴 탐지를 위한 사전 훈련된 모델 로드
face_cascade = cv2.CascadeClassifier('C:/Users/lesle/flask/.venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# 영상 파일 읽기
cap = cv2.VideoCapture('dance.mp4')

# 잘라낸 얼굴을 저장할 리스트
cropped_faces = []

while True:
    # 프레임별로 영상 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 탐지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 탐지된 얼굴에 사각형 그리기 및 얼굴 잘라내기
    for (x, y, w, h) in faces:
        # 얼굴 부분 잘라내기
        cropped_face = frame[y:y+h, x:x+w]
        # 잘라낸 얼굴을 리스트에 저장
        cropped_faces.append(cropped_face)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 잘라낸 얼굴 이미지 처리(첫 번째 이미지)
if len(cropped_faces) > 0:
    cv2.imshow("First Cropped Face", cropped_faces[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
