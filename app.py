from flask import Flask,jsonify
from connection import s3_connection
from config import BUCKET_NAME

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/image', methods=['POST'])
def test_image():
    s3 = s3_connection()
    try:
        with open('dd.jpeg', 'rb') as image_file:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Body=image_file,
                Key='dd.jpeg',
                ContentType='image/jpeg'
            )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')

