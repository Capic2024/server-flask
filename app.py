from flask import (Flask, request, send_file)
import werkzeug
import os

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
def video():  # put application's code here
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    filename = werkzeug.utils.secure_filename(file.filename)

    # 파일 저장 경로 설정
    save_path = os.path.join("your/storage/path/", filename)
    file.save(save_path)

    # 파일 처리 로직을 여기에 작성 (예시에서는 단순 저장만 하고 있음)

    # 처리한 파일을 Spring Boot로 반환
    return send_file(save_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
