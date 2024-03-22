from flask import (Flask, jsonify)
from flask.cli import load_dotenv
import boto3
# from connection import s3_connection
import os
import logging

app = Flask(__name__)

load_dotenv()


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/test', methods=['POST'])
def test():  # put application's code here
    return 'post test'


@app.route('/test2', methods=['GET'])
def test2():  # put application's code here
    return 'post test2'


@app.route('/image', methods=['POST'])
def test_image():
    try:
        # 환경 변수 로깅
        logging.info("AWS_ACCESS_KEY_ID Type: %s", type(os.environ.get('AWS_ACCESS_KEY')))
        logging.info("AWS_SECRET_ACCESS_KEY Type: %s", type(os.environ.get('AWS_SECRET_KEY')))
        logging.info("REGION_NAME Type: %s", type(os.environ.get('REGION_NAME')))
        logging.info("BUCKET_NAME Type: %s", type(os.environ.get('BUCKET_NAME')))

        client = boto3.client('s3',
                              aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
                              aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
                              region_name=os.environ.get('REGION_NAME')
                              )

        file_name = 'dd.jpeg'  # 업로드할 파일 이름
        bucket = os.environ.get('BUCKET_NAME')  # 버켓 주소
        key = 'dd.jpeg'  # s3 파일 이미지

        # 파일 및 버킷 정보 로깅
        logging.info("file_name Type: %s, bucket Type: %s, key Type: %s", type(file_name), type(bucket), type(key))

        client.upload_file(file_name, bucket, key)  # 파일 저장
        return jsonify({'success': True})

    except TypeError as e:
        # TypeError 발생 시, 오류 로깅
        logging.error("TypeError 발생: %s", str(e))
        return jsonify({'error': "TypeError 발생: " + str(e)})
    except Exception as e:
        # 기타 예외 발생 시, 오류 로깅
        logging.error("일반 오류 발생: %s", str(e))
        return jsonify({'error': "일반 오류 발생: " + str(e)})


# s3 = s3_connection()
# try:
#     with open('dd.jpeg', 'rb') as image_file:
#         s3.put_object(
#             Bucket=os.environ.get('BUCKET_NAME'),
#             Body=image_file,
#             Key='dd.jpeg',
#             ContentType='image/jpeg'
#         )
#     return jsonify({'success': True})
# except Exception as e:
#     logging.error("Error uploading image: %s",str(e))
#     return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
