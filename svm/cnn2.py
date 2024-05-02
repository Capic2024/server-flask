import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# 특징 추출 함수
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    resized_img = cv2.resize(image, (64, 64))
    fd = hog(resized_img, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), visualize=False, feature_vector=True)
    return fd

def load_images_from_directory(directory, sample_size=None):
    features = []
    labels = []
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]
    if sample_size:
        image_files = np.random.choice(image_files, size=sample_size, replace=False)

    total_images = len(image_files)
    for index, file in enumerate(image_files):
        feat = extract_hog_features(file)
        if feat is not None:
            features.append(feat)
            labels.append(1)  # 모든 이미지에 대해 '얼굴' 레이블(1) 부여
        print(f"Processed {index + 1}/{total_images} images.")

    return features, labels

# 데이터셋 경로 설정
celeba_dir = './img_align_celeba'  # CelebA 이미지 디렉토리
wider_face_dir = './output'  # WIDER FACE 이미지 디렉토리

# Load data
celeba_features, celeba_labels = load_images_from_directory(celeba_dir, sample_size=500)
wider_features, wider_labels = load_images_from_directory(wider_face_dir, sample_size=500)

# 데이터 합치기
features = np.vstack((celeba_features, wider_features))
labels = np.hstack((celeba_labels + wider_labels))
print('데이터 합치기 끝')

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
print('데이터 분할 끝')

# 그리드 서치를 이용한 SVM 튜닝 및 훈련
parameters = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(svm.SVC(), parameters, refit=True, verbose=2, cv=3)
grid_search.fit(X_train, y_train)
print('튜닝 및 훈련 끝')

# 모델 평가
y_pred = grid_search.predict(X_test)
print("Best parameters found:", grid_search.best_params_)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 모델 저장
joblib.dump(grid_search.best_estimator_, 'svm_face_recognition2.pkl')
