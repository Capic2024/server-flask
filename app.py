from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/test', methods=['POST'])
def hello_world():  # put application's code here
    return 'post test'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
