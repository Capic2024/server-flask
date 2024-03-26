import io

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
def video():  # put application's code here
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    # 메모리 상에서 파일 처리
    # 예시로, 파일을 메모리 상에서 읽고 바로 반환합니다.
    # 실제로는 이 부분에서 파일을 처리하는 로직을 구현합니다.
    file_stream = io.BytesIO()
    file.save(file_stream)
    file_stream.seek(0)  # 파일 포인터를 처음으로 이동

    # 처리된 파일 스트림을 바로 응답으로 반환
    return send_file(
        file_stream,
        as_attachment=True,
        attachment_filename=file.filename,
        mimetype=file.content_type
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
