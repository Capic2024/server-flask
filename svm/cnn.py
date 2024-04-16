import os
import cv2
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 이미지 로딩 및 전처리 함수
def load_images_from_directory(directory, size=(64, 64)):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Image not loaded properly from {img_path}")
                continue
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    # Convert the list of images to a 4D numpy array
    images = np.array(images)  # This should create an array of shape (num_images, 64, 64, 3)
    return images

# 데이터셋 로드
# celeba_images = load_images_from_directory('img_align_celeba')
images = load_images_from_directory('output')

print("Data Loaded Successfully")
# print(f"Number of CelebA images: {len(celeba_images)}")
print(f"Number of Output images: {len(images)}")

# 데이터 결합 및 분할
all_images = np.concatenate(( images), axis=0)
X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)

data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
train_generator = data_generator.flow(X_train, batch_size=32)



print("Data split into training and test sets.")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 모델 구축 함수
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Assuming binary classification for simplicity
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model((64, 64, 3))

# 데이터 증강
data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
train_generator = data_generator.flow(X_train, batch_size=32)

# 모델 학습 (임시로 labels 만들기 - 실제 사용 시 적절한 labels 준비 필요)
dummy_labels = np.random.randint(2, size=len(X_train))
dummy_labels = to_categorical(dummy_labels, num_classes=2)
# Assuming you have a 'model' defined
model.fit(train_generator, epochs=10, validation_data=(X_test, np.random.randint(0, 2, size=(len(X_test), 1))))
print("Model training complete.")

# 모델 저장
model.save('face_recognition_model.h5')
print("Model saved successfully.")
