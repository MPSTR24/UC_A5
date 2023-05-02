from flask import Flask, request, jsonify
import json
import os
import numpy as np
import pickle
from ntfy import ntfy

app = Flask(__name__)

ROOT_DIR = os.getcwd()

rocket_classifier = pickle.load(open("./rocket_model.pkl", "rb"))

notify = ntfy()

@app.route('/test_predict', methods=['POST'])

def test_predict():
    data = request.get_json()

    data = np.array(data)

    prediction = rocket_classifier.predict(data)

    notify.notify(prediction[0])


    print(prediction)

    # response = [prediction[0], prediction_probabilties[0]]

    return prediction[0]

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()

    data = np.array(data)

    sensor_data = data[0:len(data)-1]

    user_data = data[-1]


    # prediction = rocket_classifier.predict(sensor_data)

    print(sensor_data)
    # print(prediction)

    # response = [prediction[0], prediction_probabilties[0]]

    return "success"



if __name__ == '__main__':
    app.run()