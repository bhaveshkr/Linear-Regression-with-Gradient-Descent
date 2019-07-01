import sys
import json
import numpy as np

from flask import Flask, request
from util import util
from model import model

model = model()
app = Flask(__name__)

@app.route("/")
def hello():
    version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    message = "Hello World from Flask in a Docker container running Python {} with Meinheld and Gunicorn on Alpine (default)".format(
        version
    )
    return message

@app.route('/train')
def train():
    X, y = util.preprocess()
    return model.train(X, y)

@app.route('/test', methods=['POST'])
def predict():
    data = json.loads(request.data)
    X = util.json_to_np(data)
    return model.predict(X)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')