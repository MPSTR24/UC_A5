import os
from sktime.datasets import load_from_tsfile
from sktime.classification.kernel_based import RocketClassifier
# from sktime.utils import mlflow_sktime
import numpy as np
import pandas as pd
import pickle
import json


def main():

    ROOT_DIR = os.getcwd()

    print(ROOT_DIR)

    DATASET_PATH = os.path.join(ROOT_DIR, 'dataset', 'safety_recognition', 'safety_recognition.ts',)

    rocket_classifier = RocketClassifier(num_kernels=10000, rocket_transform="minirocket")


    X, y = load_from_tsfile(DATASET_PATH, return_data_type="numpy3D")


    rocket_classifier.fit(X, y)

    pickle.dump(rocket_classifier, open("./server/rocket_model.pkl", "wb"))


    # mlflow_sktime.save_model(sktime_model=rocket_classifier,path="./server/rocket_model") 



if __name__ == "__main__":
    main()
