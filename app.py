from flask import Flask

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
    return 'post test3'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
