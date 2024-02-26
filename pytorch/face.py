import cv2
import numpy as np
import face_recognition

#이미지 비교를 위해 2개 이미지 불러옴
standard = face_recognition.load_image_file('./wonbin1.jpg') #인코딩할 이미지
standard = cv2.cvtColor(standard,cv2.COLOR_BGR2RGB) #rgb로 변환
#test = face_recognition.load_image_file('./juk.jpg')
#test = face_recognition.load_image_file('./wonbin2.jpg')
#test = face_recognition.load_image_file('./goognyoo.png')
test = face_recognition.load_image_file('./gongyoo2.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

#얼굴 위치 같게 만들어줌
faceLocStandard = face_recognition.face_locations(standard)[0] #감지할 얼굴 인코딩(이미지 처리를 위해 이미지 하나하나 픽셀값을 코드로 바꿔줌)
encodeStandard = face_recognition.face_encodings(standard)[0] #첫번째 요소 가져옴
cv2.rectangle(standard, (faceLocStandard[3], faceLocStandard[0]), (faceLocStandard[1], faceLocStandard[2]),(255,0,255),2) #감지된 얼굴 위치 확인을 위한 사각형 그려주기

faceLocTest=face_recognition.face_locations(test)[0]
encodeTest = face_recognition.face_encodings(test)[0]   #Test이미지에 대한 첫번째 요소만 가져오기
cv2.rectangle(test,(faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#두 얼굴 간의 오차 파악
result = face_recognition.compare_faces([encodeStandard],encodeTest) #공유 얼굴이랑 원빈 얼굴 비교
faceDis = face_recognition.face_distance([encodeStandard],encodeTest) # 유사성 알아내기 (-> 얼굴 간의 오차)
print(result, faceDis) #같은 얼굴이면 True, 다른 이미지이면 False

# 이미지에 대한 결과랑 유사성을 Test 이미지에 명시해주기, round()는 유사성을 소수점 둘째짜리로 반올림 한다는 뜻
cv2.putText(test, f'{result} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Standard', standard)  # 인코딩 이미지 불러오기
cv2.imshow('Test', test)  # Test 이미지 불러오기
cv2.waitKey(0)

