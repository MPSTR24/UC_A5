import os
from sktime.datasets import load_from_tsfile
from sktime.classification.kernel_based import RocketClassifier
import numpy as np
import pandas as pd
import pickle


def main():

    ROOT_DIR = os.getcwd()

    print(ROOT_DIR)

    DATASET_PATH = os.path.join(ROOT_DIR, 'Data', 'datasets', "safety_recognition" 'safety_recognition.ts',)

    rocket_classifier = RocketClassifier(num_kernels=1000)


    X, y = load_from_tsfile(DATASET_PATH, return_data_type="numpy3D")


    rocket_classifier.fit(X, y)


    # pickle.dump(rocket_classifier, open("./server/rocket_model.pkl", "wb"))



    









if __name__ == "__main__":
    main()
