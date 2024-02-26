from flask import Flask,jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run()

@app.route('/sendImg',methods=['GET'])
def sendImg():
    return jsonify({'class_id':'IMGAGE'})
