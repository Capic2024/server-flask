import cv2
import numpy as np


def video(cam, net):
    while True:
        ret, img = cam.read()
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.imshow("facenet model", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Load pre-trained deep learning model for face detection
model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Capture video from webcam or file
cam = cv2.VideoCapture('video.mp4')

# Run face detection on video
video(cam, net)

# Release video capture object and close windows
cam.release()
cv2.destroyAllWindows()
