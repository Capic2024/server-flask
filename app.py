import io
import os
from pytorch import mosaic_jiyeon

from flask import (Flask, request, send_file)

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/test', methods=['POST'])
def test():  # put application's code here
    return 'post test'


@app.route('/test2', methods=['GET'])
def test2():  # put application's code here
    return '03.25 22:40'


@app.route('/video', methods=['POST'])
def handle_video():
    print('start')
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    print('file', file.filename)

    # 파일 저장 경로 지정, 여기서는 임시로 파일 이름으로 저장
    filename = os.path.join('/tmp', file.filename)
    file.save(filename)
    print('success file')

    # 여기서 파일을 처리하는 로직을 추가할 수 있습니다.
    # 예를 들어, 처리된 파일을 다시 클라이언트에게 보낼 수 있습니다.
    # 이 예시에서는 단순히 저장된 파일을 그대로 반환합니다.
    image_paths = ["img.png", "img_1.png", "goognyoo.png"]
    print('start mosaic')
    output_video_path = mosaic_jiyeon.mosaic(filename, image_paths)

    return send_file(output_video_path, mimetype='video/mp4', as_attachment=True)


@app.route('/image', methods=['POST'])
def image_test():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # 파일 저장 경로 지정, 여기서는 임시로 파일 이름으로 저장
    filename = os.path.join('/tmp', file.filename)
    file.save(filename)

    return send_file(filename, mimetype='image.jpeg', as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
