import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# WideFace 데이터셋 클래스 정의
class WideFaceDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 형식으로 이미지를 읽으므로 RGB로 변환

        if self.transform:
            image = self.transform(image)

        return image

# WideFace 데이터셋 변환 함수 정의
transform = transforms.Compose([
    transforms.ToPILImage(),  # OpenCV 이미지를 PIL 이미지로 변환
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지를 정규화
])

# WideFace 데이터셋 인스턴스 생성
images_dir = './output'
dataset = WideFaceDataset(images_dir, transform=transform)

# DataLoader 생성
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# WideResNet 모델 정의, 최신 이미지넷 가중치 사용
model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # 출력 레이어 수정 (이진 분류를 위해 출력 뉴런 수를 1로 설정)

# 모델 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 5
model.to(device)
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        labels = torch.ones(images.size(0), 1, device=device)  # WideFace 데이터셋은 얼굴 이미지이므로 레이블은 모두 1로 설정

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 모델 저장
output_model_path = 'wideface_recognition_model.pth'
torch.save(model.state_dict(), output_model_path)
print(f'Model saved to {output_model_path}')
