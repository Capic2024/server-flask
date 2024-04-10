import cv2
import face_recognition

# 얼굴 이미지를 추출하는 함수 정의
def extract_faces_from_video(video_path):
    face_images = []  # 추출된 얼굴 이미지를 저장할 리스트

    # 비디오를 읽기 위한 VideoCapture 객체 생성
    video_capture = cv2.VideoCapture(video_path)

    # 비디오의 각 프레임을 순회
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        # 현재 프레임에서 얼굴 위치를 찾기
        face_locations = face_recognition.face_locations(frame)

        # 각 얼굴에 대한 이미지를 리스트에 추가
        for top, right, bottom, left in face_locations:
            face_image = frame[top:bottom, left:right]
            face_images.append(face_image)

    video_capture.release()

    return face_images # return

# 비디오 파일 경로를 매개변수로 전달
video_path = 'path_to_your_video.mp4'  # 원하는 비디오 파일 경로 입력
face_images = extract_faces_from_video(video_path)

# 결과를 확인. 추출된 첫 번째 얼굴 이미지를 표시.
if face_images:
    cv2.imshow('First extracted face', face_images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No faces were found in the video.")
