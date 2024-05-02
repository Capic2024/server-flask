import os

import cv2
import numpy as np
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

# 데이터셋 클래스 정의
class CelebADataset(Dataset):
    def __init__(self, image_dir, annot_file, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annot_file, delim_whitespace=True, header=1)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.image_dir, img_id))
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            img = self.transform(img)

        return (img, y_label)

# 모델 정의
class FaceRecognitionModel(nn.Module):
    def __init__(self):
        super(FaceRecognitionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 이미지에 대한 HOG 특징 추출 함수
def extract_hog_features(image):
    resized_img = cv2.resize(image, (64, 64))  # 이미지 크기 조정
    fd = hog(resized_img, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), visualize=False, feature_vector=True)
    return fd

def apply_mosaic(frame, x, y, w, h, factor=12):
    """Applies a mosaic to a specified rectangle in the image."""
    face_img = frame[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (w//factor, h//factor), interpolation=cv2.INTER_LINEAR)
    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = face_img
    return frame

def draw_face_boxes(frame, faces):
    """Draws green bounding boxes around detected faces."""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

# 비디오 처리 함수
# 비디오 처리 함수
def mosaic(video_path, image_paths):
    # MTCNN으로 얼굴 감지 모델 초기화
    mtcnn = MTCNN()

    # InceptionResnetV1로 얼굴 인식 모델 초기화
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # 얼굴 인코딩을 저장할 리스트
    encodings = []

    # 각 이미지에서 얼굴을 인코딩하여 리스트에 추가
    for image_path in image_paths:
        # 이미지 파일 로드
        image = cv2.imread(image_path)

        # MTCNN으로 얼굴 감지
        boxes, probs = mtcnn.detect(image)

        if boxes is not None:
            # 감지된 얼굴 중에서 가장 큰 얼굴 선택
            largest_face = boxes[probs.argmax()]

            x1, y1, x2, y2 = largest_face.astype(int)

            # 얼굴 영역 추출
            face_roi = image[y1:y2, x1:x2]

            # PIL 이미지로 변환
            face_image = transforms.ToPILImage()(face_roi)

            face_image_tensor = transforms.ToTensor()(face_image)

            # 얼굴 인코딩
            encoding = resnet(face_image_tensor.unsqueeze(0))
            encodings.append(encoding)

    # 모자이크 처리할 사이즈 정의
    block_size = 10

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 결과 동영상 파일 생성
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = os.path.join('tmp', 'output_video.mp4')
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 동영상 프레임마다 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MTCNN을 사용하여 객체 감지 및 얼굴 검출
        boxes, _ = mtcnn.detect(frame)

        threshold = 0.6

        # 감지된 얼굴에 모자이크 처리
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)

                # 얼굴 영역 추출
                face_roi = frame[y1:y2, x1:x2]

                # 얼굴 이미지 크기 조정
                face_roi_resized = cv2.resize(face_roi, (224, 224))

                # PIL 이미지로 변환
                face_image = transforms.ToPILImage()(face_roi_resized)

                # PIL 이미지를 Tensor로 변환
                face_image_tensor = transforms.ToTensor()(face_image)

                # 얼굴 인코딩
                encoding = resnet(face_image_tensor.unsqueeze(0))

                # 특정 사람과의 얼굴 일치 여부 확인
                match = False
                for enc in encodings:
                    # 각 얼굴의 특징 벡터를 비교하여 유사성 판단
                    similarity = torch.nn.functional.cosine_similarity(encoding, enc, dim=1)
                    if similarity > threshold:  # 유사성이 임계값보다 크면 얼굴이 일치한다고 판단
                        match = True
                        break

                if not match:  # 특정 사람과 일치하지 않는 경우에만 모자이크 처리
                    blurred_face = cv2.resize(face_roi, (block_size, block_size))
                    blurred_face = cv2.resize(blurred_face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                    frame[y1:y2, x1:x2] = blurred_face

        # 모자이크 처리된 프레임 결과 동영상에 추가
        out.write(frame)

    # 리소스 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path


# 사용자가 매칭할 얼굴 이미지들의 경로
user_images = ['./Gongyoo/image1.jpg', './Gongyoo/image2.png', './Gongyoo/img.png']

# 비디오 파일 경로
video_path = './cutVideo.mp4'

# 모델 파일 경로
model_path = './face_recognition_model.pth'

# 비디오 처리 함수 호출
mosaic(video_path, user_images)
