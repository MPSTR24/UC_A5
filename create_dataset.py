import pandas as pd
import numpy as np
import os
from sktime.datasets._data_io import write_ndarray_to_tsfile


def create_dataset(path_to_save):
    CURRENT_PATH = os.getcwd()
    DATA_DIR = os.path.join(CURRENT_PATH, "data_instances")

    X = []
    y = []

    for activity in os.listdir(DATA_DIR):

        current_activity_directory = os.path.join(DATA_DIR, activity)

        for data_instance in os.listdir(current_activity_directory):

            class_label = data_instance.split("_")
            data_instance = pd.read_csv(
                os.path.join(current_activity_directory, data_instance)
            )
            data_instance = data_instance.drop(columns=["timestamp", "label"])
            data_instance = np.transpose(data_instance)

            X.append(data_instance)
            y.append(class_label[0])

    X = np.asarray(X)
    y = np.asarray(y)

    print(np.shape(X))
    print(np.shape(y))

    # """ get the unique class labels """
    class_labels = set(y)

    write_ndarray_to_tsfile(
        data=X,
        path=f"{path_to_save}",
        problem_name="safety_recognition",
        class_label=class_labels,
        class_value_list=y,
        equal_length=True,
        series_length=100,
    )


def main():
    CURRENT_PATH = os.getcwd()
    path_to_save = os.path.join(CURRENT_PATH, "dataset")

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    create_dataset(path_to_save=path_to_save)


if __name__ == "__main__":
    main()
