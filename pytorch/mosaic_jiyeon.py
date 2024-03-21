import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data_dir = "Dataset/WIDER_train/WIDER_train/images"

# WIDER FACE 데이터셋 로드 및 전처리
def load_data(data_dir):
    images = []  # 이미지 파일 경로를 저장할 리스트

    # 데이터 디렉토리 내의 각 폴더를 순회
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):  # 하위 디렉토리인 경우
            # 하위 디렉토리 내의 이미지 파일을 로드
            for filename in os.listdir(subdir_path):
                if filename.endswith(".jpg"):  # 이미지 파일
                    image_path = os.path.join(subdir_path, filename)
                    images.append(image_path)

    return images


def split_dataset(images, test_size=0.2, random_state=42):
    # 이미지를 학습용, 테스트용으로 나눔
    train_images, test_images = train_test_split(images, test_size=test_size, random_state=random_state)

    return train_images, test_images

# 데이터 전처리

def preprocess_data(images, image_size=(224, 224)):
    preprocessed_images = []  # 전처리된 이미지를 저장할 리스트

    for image_path in images:
        # 이미지 크기 조정 및 정규화
        image = load_img(image_path, target_size=image_size)
        image = img_to_array(image) / 255.0
        preprocessed_images.append(image)

        # 레이블 포맷 변환
        # 여기서 label은 얼굴의 위치와 크기를 나타내는 정보일 것입니다.
        # 필요에 따라 레이블을 적절한 형식으로 변환하는 코드 작성

    return np.array(preprocessed_images)

# 손실 함수 정의
def custom_loss(y_true, y_pred):
    # 손실 함수를 정의하는 코드 작성
    loss = BinaryCrossentropy()(y_true, y_pred)
    return loss

# 최적화 알고리즘 선택
# 실수값 할당
#optimizer = Adam(learning_rate=0.001)
optimizer = Adam(learning_rate=0.001)

# 학습 데이터 로드 및 전처리
images = load_data(data_dir)

dataset_size = len(images)
print("데이터셋 크기:", dataset_size)

preprocessed_images = preprocess_data(images)

# 데이터셋 나누기
train_images, test_images = split_dataset(preprocessed_images)


# 클래스 레이블링 (예시로 모든 이미지가 얼굴이 있는 것으로 가정)
train_labels = np.ones(len(train_images))  # 얼굴이 있는 이미지에 대한 레이블 (1)
test_labels = np.ones(len(test_images))  # 얼굴이 있는 이미지에 대한 레이블 (1)

# 클래스 레이블을 one-hot 인코딩
train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

# VGG16 모델 로드
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fully Connected 레이어 추가
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # 2개의 클래스에 대한 예측이므로 클래스 수는 2로 설정

# 모델 구성
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accauracy'])

# 학습
history = model.fit(train_images, train_labels, epochs=1, batch_size=8, validation_data=(test_images, test_labels))

# 모델 평가
loss, accuracy = model.evaluate(test_images, test_labels)
print("손실:", loss)
print("정확도:", accuracy)