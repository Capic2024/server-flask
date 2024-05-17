import cv2
import os
from deepface import DeepFace
from retinaface import RetinaFace
import torch

def mosaic(video_path, image_paths):
    output_video_path = os.path.join('tmp', 'output.mp4')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (frame_width, frame_height))

    model_name = "Facenet"

    threshold = 0.45
    not_threshold = 0.47

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./pytorch/best.pt')

    faces_dir = os.path.join('tmp', 'faces')
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)


    face_count = 0
    current_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        #detections = RetinaFace.detect_faces(img_path=frame)

        print('감지 시작')
        for result in results.xyxy[0]:
            # print(face_id)
            # x1, y1, x2, y2 = [int(v) for v in detections[face_id]['facial_area']]
            if result[5] == 0:
                x1, y1, x2, y2 = result[:4].int().tolist()
                # 얼굴 영역 추출
                face_image = frame[y1:y2, x1:x2]
                matched = False
                for ref_face in image_paths:
                    result = DeepFace.verify(img1_path=face_image, img2_path=ref_face, model_name=model_name,
                                             detector_backend='retinaface', enforce_detection=False)
                    print('end verify')
                    distance = result['distance']
                    verified = result['verified']
                    if 0.46 >= distance >= 0.45:
                        verified_str = 'Different'
                        distance_str = '(%.4f <= %.4f)' % (distance, threshold)
                        print(verified_str, distance_str)
                        face = cv2.resize(face_image, (10, 10))
                        face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                        frame[y1:y2, x1:x2] = face

                    if distance < threshold:
                        break

                    if distance > not_threshold:
                        face = cv2.resize(face_image, (10, 10))
                        face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                        frame[y1:y2, x1:x2] = face
                        break

                face_filename = f"face_{face_count}.jpg"
                face_filepath = os.path.join(faces_dir, face_filename)
                cv2.imwrite(face_filepath, face_image)
                face_count += 1
        current_frame_count += 1
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path