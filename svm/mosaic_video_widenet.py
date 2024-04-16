import cv2
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# 모델과 이미지 경로 설정
model_path = 'wideface_recognition_model.pth'
user_images = ["image1.jpg", "image2.jpg"]  # 사용자 이미지 경로
video_path = 'input_video.mp4'  # 입력 비디오 경로
output_path = 'output_video.mp4'  # 출력 비디오 경로

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='local', force_reload=True)
mtcnn = MTCNN(keep_all=True, device='cuda')

# 이미지를 텐서로 변환하는 함수
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# 이미지 파일에서 얼굴 특징 벡터를 추출하는 함수
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform_image(image)
    features = model(image.unsqueeze(0).to('cuda')).detach().cpu().numpy()
    return features

# 이미지 파일들의 얼굴 특징 벡터를 추출하여 저장
user_features = [extract_features(image) for image in user_images]

# 비디오 캡처
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 비디오 프레임마다 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5를 사용하여 객체 감지
    results = model(frame)

    # 감지된 얼굴에 대해 얼굴 특징 벡터를 추출하여 매칭 여부 확인
    for result in results.xyxy[0]:
        if result[5] == 0:  # 클래스 인덱스가 0일 때(사람 얼굴을 의미하는 클래스)
            x1, y1, x2, y2 = result[:4].int().tolist()
            face_roi = frame[y1:y2, x1:x2]  # 얼굴 영역 추출

            # PIL 이미지로 변환하여 MTCNN을 사용하여 얼굴 감지 및 특징 벡터 추출
            pil_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            boxes, _ = mtcnn.detect(pil_image)
            if boxes is not None:
                for box in boxes:
                    x, y, w, h = box.astype(int)
                    face_image = pil_image.crop((x, y, x+w, y+h))
                    face_features = model(transform_image(face_image).unsqueeze(0).to('cuda')).detach().cpu().numpy()

                    # 각 얼굴의 특징 벡터를 비교하여 매칭 여부 판단
                    match = False
                    for user_feature in user_features:
                        similarity = np.dot(face_features, user_feature.T) / (np.linalg.norm(face_features) * np.linalg.norm(user_feature))
                        if similarity > 0.8:  # 유사성이 임계값보다 크면 얼굴이 일치한다고 판단
                            match = True
                            break

                    if match:  # 매칭되는 얼굴은 모자이크 처리하지 않음
                        continue

                    # 매칭되지 않는 얼굴은 모자이크 처리
                    blurred_face = cv2.resize(face_roi, (10, 10))
                    blurred_face = cv2.resize(blurred_face, (w, h), interpolation=cv2.INTER_AREA)
                    frame[y1:y2, x1:x2] = blurred_face

    # 모자이크 처리된 프레임 결과 동영상에 추가
    out.write(frame)

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")
