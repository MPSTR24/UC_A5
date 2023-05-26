import os
from sktime.datasets import load_from_tsfile
import requests
import json


def main():

    ROOT_DIR = os.getcwd()

    print(ROOT_DIR)
    DATASET_PATH_TEST = os.path.join(ROOT_DIR,'dataset', 'safety_recognition', 'safety_recognition.ts')

    X_test, y_test = load_from_tsfile(DATASET_PATH_TEST, return_data_type="numpy3D")


    headers = {'Content-type': 'application/json'}

    response = requests.post('http://127.0.0.1:5000/test_predict', data=json.dumps(X_test[62:63].tolist()), headers=headers)
    print(response.text)


if __name__ == '__main__':
    main()