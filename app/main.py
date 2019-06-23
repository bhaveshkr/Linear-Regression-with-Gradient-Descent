import sys
import json

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


@app.route('/test')
def predict():
    data = request.data
    X = json.loads(data)
    return model.predict(X)