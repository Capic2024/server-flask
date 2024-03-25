from flask import (Flask, jsonify, send_file)
from dotenv import load_dotenv
import boto3
# from connection import s3_connection
import os
import logging

app = Flask(__name__)
load_dotenv()
logging.basicConfig(level=logging.ERROR)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/test', methods=['POST'])
def test():  # put application's code here
    return 'post test'


@app.route('/test2', methods=['GET'])
def test2():  # put application's code here
    return '03.25 22:40'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
