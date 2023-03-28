import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():

    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "spliced")

    for activity in os.listdir(DATA_PATH):

        CURRENT_ACIVITY = os.path.join(DATA_PATH, activity)
        DATA_INSTANCES = os.path.join(CURRENT_PATH, "data_instances", activity)

        if not os.path.exists(DATA_INSTANCES):
            os.makedirs(DATA_INSTANCES)

        instance_num = 1

        for recording in os.listdir(CURRENT_ACIVITY):

            data_instance = pd.read_excel(os.path.join(CURRENT_ACIVITY, recording, "data_selected.xlsx"))

            if np.shape(data_instance) == (20, 8):
                data_instance.to_csv(f'./data_instances/{activity}/{activity}_{instance_num}.csv', index = False)
                instance_num += 1



        


if __name__ == "__main__":
    main()