import cv2
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from collections import defaultdict

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

#from model_definition import SideFaceDetector  # 학습된 모델이 정의된 클래스 또는 스크립트

# 모델 불러오기
# model = SideFaceDetector()  # 학습된 모델이 정의된 클래스 또는 스크립트를 불러옵니다.
# model.load_state_dict(torch.load('side_face_detector.pth'))  # 학습된 모델의 가중치를 불러옵니다.
# model.eval()  # 평가 모드로 설정

haarcascades_path = cv2.data.haarcascades
print("하르 캐스케이드 파일 디렉토리:", haarcascades_path)

fps = 30.0  # 프레임 속도 설정
codec = cv2.VideoWriter_fourcc(*'XVID')
output_size = (640, 480)  # 저장할 동영상의 크기 설정

# VideoWriter 객체 생성
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

model = fasterrcnn_resnet50_fpn(pretrained=True, weights=None)

# model = fasterrcnn_resnet50_fpn(pretrained=True, weights='torchvision://fasterrcnn_resnet50_fpn')


model.eval()

# 전처리 함수 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

video_path = "work.mp4"

# 동영상 파일 열기
cap = cv2.VideoCapture(video_path)

# 얼굴 크기와 등장 시간 기록을 위한 딕셔너리 초기화
face_sizes = defaultdict(int)

# 동영상에서 얼굴 감지 및 처리
while cap.isOpened():
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if face_cascade.empty():
        print("하르 캐스케이드 분류기를 로드하는 데 문제가 발생했습니다.")
    else:
        print("하르 캐스케이드 분류기가 정상적으로 로드되었습니다.")
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 각 얼굴의 크기와 등장 시간 기록
    for (x, y, w, h) in faces:
        face_sizes[(x, y, w, h)] += 1

    # 화면에 얼굴 박스 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    out.write(frame)

    # 화면에 표시
    cv2.imshow('Face Detection', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 동영상 파일 닫기
out.release()
cap.release()
cv2.destroyAllWindows()