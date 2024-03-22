from flask import (Flask, jsonify)
# from connection import s3_connection
# import os

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/test', methods=['POST'])
def test():  # put application's code here
    return 'post test'

@app.route('/test2', methods=['GET'])
def test2():  # put application's code here
    return 'post test2'

@app.route('/test3', methods=['GET'])
def test3():  # put application's code here
    return 'post test3 again2'

@app.route('/test4', methods=['GET'])
def test4():  # put application's code here
    return 'post test4'


# @app.route('/image', methods=['POST'])
# def test_image():
#     s3 = s3_connection()
#     try:
#         with open('dd.jpeg', 'rb') as image_file:
#             s3.put_object(
#                 Bucket=os.environ.get('BUCKET_NAME'),
#                 Body=image_file,
#                 Key='dd.jpeg',
#                 ContentType='image/jpeg'
#             )
#         return jsonify({'success': True})
#     except Exception as e:
#         return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
