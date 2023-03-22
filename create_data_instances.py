import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def raw_data(raw_accel_data, raw_gyro_data, activity, segment_size):

    """
    combine the accelerometer and gyroscope data for activities and resampling to a frequency of every 100ms
    also interpolating any missing values from the sensors

    """

    # rename axis so there is no clash when joining tables
    raw_accel_data = raw_accel_data.rename(columns={"x": "ax", "y": "ay", "z": "az"})
    raw_gyro_data = raw_gyro_data.rename(columns={"x": "gx", "y": "gy", "z": "gz"})

    combined_data = pd.DataFrame(
        columns=["timestamp", "label", "ax", "ay", "az", "gx", "gy", "gz"]
    )

    CURRENT_PATH = os.getcwd()
    LABEL_PATH = os.path.join(CURRENT_PATH, "data_instances", activity)
    if not os.path.exists(LABEL_PATH):
        os.makedirs(LABEL_PATH)

    instance_num = 1

    # get individual recordings and loop over them
    session_id = raw_accel_data["session_id"].unique()

    for i in range(len(session_id)):

        current_session_id = session_id[i]
        label = raw_accel_data["label"].unique()

        accel_data = raw_accel_data.loc[
            raw_accel_data["session_id"] == session_id[i]
        ].reset_index(drop=True)
        gyro_data = raw_gyro_data.loc[
            raw_gyro_data["session_id"] == session_id[i]
        ].reset_index(drop=True)

        # convert timestamp columns to datetime datatype
        accel_data["timestamp"] = pd.to_datetime(
            accel_data["timestamp"], unit="ns", origin="unix"
        ).dt.floor("1ms")
        gyro_data["timestamp"] = pd.to_datetime(
            gyro_data["timestamp"], unit="ns", origin="unix"
        ).dt.floor("1ms")

        # set timestamp column as index
        accel_data.set_index("timestamp", inplace=True)
        gyro_data.set_index("timestamp", inplace=True)

        # resample both dataframes to a frequency of 100ms
        df_acc_resampled = accel_data.resample("100ms").mean()
        df_gyro_resampled = gyro_data.resample("100ms").mean()

        # merge the two resampled dataframes using the closest timestamp
        df_combined = pd.merge_asof(df_acc_resampled, df_gyro_resampled, on="timestamp")

        # interpolate any missing values from merging of both sensors using linear methods
        combined_timestamps = df_combined["timestamp"]
        interpolated_values = df_combined[
            ["ax", "ay", "az", "gx", "gy", "gz"]
        ].interpolate()

        data_label = pd.Series(
            [label[0] for i in range(len(combined_timestamps))], name="label"
        )

        result = pd.concat(
            [combined_timestamps, data_label, interpolated_values], axis=1
        )


        result = result.drop(result.index[range(20)]).reset_index(drop=True)
        result = result[:len(result)-20]


        for j in range(0, len(result), segment_size):
            print(j, j+segment_size)
            segment = result.iloc[j:j+segment_size]

            segment.to_csv(f'./data_instances/{label[0]}/{label[0]}_{instance_num}.csv', index = False)
            instance_num += 1

        print("session finished " + str(i))
        print("***************************")

        # print(result)

        # print(combined_data)
        combined_data = pd.concat([combined_data, result], ignore_index=True)
        
        combined_data['timestamp'] = np.arange(0, (len(combined_data))*100, 100)
        # print(np.shape(timestamps))

        # print(result)
        # print(np.array(result))

    # print(combined_data)

    # combined_data.to_csv('./combined_data.csv', index = False)


def main():

    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "data")


    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, "data")
    DATA_INSTANCE_PATH = os.path.join(CURRENT_PATH, "data_instances")
    if not os.path.exists(DATA_INSTANCE_PATH):
        os.makedirs(DATA_INSTANCE_PATH)

    for activity in os.listdir(DATA_PATH):
        print(activity)  
        
        accel_data = pd.read_excel(os.path.join(DATA_PATH, activity, "accelerometer.xlsx"))
        gyro_data = pd.read_excel(os.path.join(DATA_PATH, activity, "gyroscope.xlsx"))
        raw_data(accel_data, gyro_data, activity, 20)


if __name__ == "__main__":
    main()
