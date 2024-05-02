import cv2
import os

# 이미지와 .txt 파일이 저장된 디렉토리 경로 설정
images_dir = './images'
annotations_dir = './annotations'

# .txt 파일 로드
txt_file_path = os.path.join(annotations_dir, 'wider_face_train_bbx_gt.txt')

# 저장된 이미지 파일 경로에 따라 파일을 순회
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

i = 0
while i < len(lines):
    file_path = lines[i].strip()
    if file_path.endswith('.jpg'):  # 이미지 파일 경로 확인
        image_path = os.path.join(images_dir, file_path)
        i += 1
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found at {image_path}")
            continue

        num_faces = int(lines[i].strip())  # 해당 이미지의 바운딩 박스 개수
        i += 1

        for _ in range(num_faces):
            x, y, w, h = map(int, lines[i].strip().split()[:4])
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            i += 1

        # 변형된 이미지 저장
        output_path = './output'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, os.path.basename(file_path)), image)
    else:
        i += 1

print("Processing complete.")
