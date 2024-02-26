import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
#import utils
import os

import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

class MultiPIEDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['frontal', 'side']  # 클래스 정의 (정면, 옆모습)
        self.images = []
        self.labels = []

        # 데이터셋 이미지 파일과 레이블 파일을 로드하는 코드 추가
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(root, filename)
                    if 'straight' in img_path:  # 정면 이미지
                        self.images.append(img_path)
                        self.labels.append(0)  # 정면 클래스 레이블: 0
                    elif 'left' in img_path or 'right' in img_path:  # 옆모습 이미지
                        self.images.append(img_path)
                        self.labels.append(1)  # 옆모습 클래스 레이블: 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

# Multi-PIE 데이터셋의 경로와 전처리 방법을 설정
data_dir = 'path/to/multi_pie_dataset'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화
])

# 데이터셋 객체 생성
dataset = MultiPIEDataset(data_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

# 모델 초기화
model = maskrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # '사람'과 '배경' 클래스
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# GPU를 사용할 수 있는 경우 모델을 GPU로 이동합니다.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 옵티마이저 및 학습률 스케쥴러 설정
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 모델 학습
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    # 학습
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # 학습률 스케쥴링
    lr_scheduler.step()
    # 모델 저장
    torch.save(model.state_dict(), os.path.join('./', 'side_face_detector.pth'))
