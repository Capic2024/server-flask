import cv2
import torch
import face_recognition


def mosaic(video_path, image_paths):
    # YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt',
                           force_reload=True)
    output_video_path = 'blurred_' + video_path
    # 특정 사람의 얼굴 이미지 로드
    # person_image = face_recognition.load_image_file("goognyoo.png")
    # person_encoding = face_recognition.face_encodings(person_image)[0]

    # 얼굴 인코딩을 저장할 리스트
    encodings = []

    # 각 이미지에서 얼굴을 인코딩하여 리스트에 추가
    for image_path in image_paths:
        # 이미지 파일 로드
        image = face_recognition.load_image_file(image_path)
        # 얼굴 인코딩
        encoding = face_recognition.face_encodings(image)
        if len(encoding) > 0:
            encodings.append(encoding[0])

    # 모자이크 처리할 사이즈 정의
    block_size = 10

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 결과 동영상 파일 생성
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 동영상 프레임마다 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5를 사용하여 객체 감지
        results = model(frame)

        # 감지된 얼굴에 모자이크 처리
        for result in results.xyxy[0]:
            if result[5] == 0:  # 클래스 인덱스가 0일 때(사람 얼굴을 의미하는 클래스)
                x1, y1, x2, y2 = result[:4].int().tolist()
                # 얼굴 영역 추출
                face_roi = frame[y1:y2, x1:x2]

                # 이미지를 RGB 형식으로 변환
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                # 얼굴 인코딩
                face_encodings = face_recognition.face_encodings(face_roi_rgb)
                if len(face_encodings) > 0:
                    encoding = face_encodings[0]
                    # 특정 사람과의 얼굴 일치 여부 확인
                    match = face_recognition.compare_faces(encodings, encoding)
                    if any(match):  # 특정 사람과 일치하지 않는 경우에만 모자이크 처리
                        # 얼굴 영역에 모자이크 처리
                        continue

                blurred_face = cv2.resize(face_roi, (block_size, block_size))
                blurred_face = cv2.resize(blurred_face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                frame[y1:y2, x1:x2] = blurred_face

        # 모자이크 처리된 프레임 결과 동영상에 추가
        out.write(frame)

        # 종료
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path
