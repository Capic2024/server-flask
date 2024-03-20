import timeit
import cv2

def video(cam, cascade):
    while True:
        startTime = timeit.default_timer()
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detectMultiScale 객체 탐지 알고리즘
        result = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))

        for box in result:
            # 좌표 추출
            x, y, w, h = box
            # 경계 상자 그리기
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)

        endTime = timeit.default_timer()
        FPS = 'fps' + str(int(1. / (endTime - startTime)))
        cv2.putText(img, FPS, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow('facenet model', img)
        # waitKey : 키 입력을 기다림
        if cv2.waitKey(1) > 0:
            break

def img(image, cascade):
    # 영상 압축
    img = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detectMultiScale 객체 탐지 알고리즘
    result = cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))

    for box in result:
        # 좌표 추출
        x, y, w, h = box
        # 경계 상자 그리기
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)

    # 사진 출력
    cv2.imshow('facenet model', img)
    cv2.waitKey(10000)


cascade_filename = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_filename)

cam = cv2.VideoCapture('video.mp4')
image = cv2.imread('goognyoo.png')

# 영상 탐지기
video(cam, cascade)
# 사진 탐지기
# imgDetector(cam,cascade)
