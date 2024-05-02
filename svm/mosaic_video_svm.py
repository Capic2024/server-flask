import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity

def extract_hog_features(image):
    resized_img = cv2.resize(image, (64, 64))  # Resize image to 64x64
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

def process_video(video_path, user_images, model_path):
    model = joblib.load(model_path)  # Load the trained model
    user_features = [extract_hog_features(cv2.imread(img, cv2.IMREAD_GRAYSCALE)) for img in user_images]

    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_features = extract_hog_features(face_img)

            # Calculate similarity
            similarities = [cosine_similarity([face_features], [uf])[0][0] for uf in user_features]
            max_similarity = max(similarities) if similarities else 0

            if max_similarity > 0.8783:  # Threshold for similarity
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle for matched faces
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # frame = apply_mosaic(frame, x, y, w, h)  # Apply mosaic to non-matched faces

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# User face image paths list
user_images = ['./Gongyoo/image1.jpg', './Gongyoo/image2.png', './Gongyoo/img.png','./Gongyoo/img_1.png', './Gongyoo/image3.png', './Gongyoo/image4.png']
# Video file path and SVM model file path
video_path = './cutVideo.mp4'
model_path = './svm_face_recognition.pkl'

# Execute video processing
process_video(video_path, user_images, model_path)
