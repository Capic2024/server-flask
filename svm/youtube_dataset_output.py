import cv2
import os
from skimage.feature import hog
import numpy as np
import joblib


def extract_hog_features(image):
    """Extract HOG features from a single image."""
    resized_img = cv2.resize(image, (64, 64))  # Resize to ensure feature size consistency
    fd = hog(resized_img, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), visualize=False, feature_vector=True)
    return fd


def detect_faces_and_extract_features(video_path, face_cascade):
    """Process a video, detect faces and extract features."""
    cap = cv2.VideoCapture(video_path)
    features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            features.append(extract_hog_features(face_img))

    cap.release()
    return features


def process_dataset(dataset_path):
    """Process all videos in the dataset."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    all_features = []
    for video_file in os.listdir(dataset_path):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(dataset_path, video_file)
            print(f"Processing {video_path}")
            video_features = detect_faces_and_extract_features(video_path, face_cascade)
            all_features.extend(video_features)

    return np.array(all_features)


# Directory where the YouTube Faces dataset videos are stored
dataset_path = '/path/to/your/dataset'
features = process_dataset(dataset_path)

# Optionally save the extracted features for later use
joblib.dump(features, 'youtube_faces_features.pkl')
