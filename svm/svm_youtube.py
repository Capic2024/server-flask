import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import cv2
from skimage.feature import hog

def load_features_from_file(file_path):
    # 파일에서 특징 데이터 로드
    return joblib.load(file_path)

def extract_hog_features_from_directory(directory):
    # 디렉토리 내 이미지로부터 HOG 특징 추출
    features = []
    for file in os.listdir(directory):
        img_path = os.path.join(directory, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            features.append(extract_hog_features(img))
    return features

def extract_hog_features(image):
    resized_img = cv2.resize(image, (64, 64))  # 이미지 크기 통일
    fd, _ = hog(resized_img, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return fd

# YouTube Faces Dataset의 사전 처리된 특징 로드
youtube_features = load_features_from_file('youtube_faces_features.pkl')
youtube_labels = np.ones(len(youtube_features))  # 특정 인물 레이블 1

# WIDER FACE 데이터셋 이미지에서 특징 추출
wider_features = extract_hog_features_from_directory('./output')
wider_labels = np.zeros(len(wider_features))  # 비특정 인물 레이블 0

# 데이터셋 결합
features = np.vstack((youtube_features, wider_features))
labels = np.hstack((youtube_labels, wider_labels))

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

# SVM 훈련
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# 모델 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 훈련된 모델 저장
joblib.dump(model, 'face_recognition_model.pkl')
