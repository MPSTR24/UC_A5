import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def raw_data(raw_accel_data, raw_gyro_data):

    """ 
    combine the accelerometer and gyroscope data for activities and resampling to a frequency of every 100ms
    also interpolating any missing values from the sensors 
    
    """

    # rename axis so there is no clash when joining tables
    raw_accel_data = raw_accel_data.rename(columns={"x": "ax", "y": "ay", "z": "az"})
    raw_gyro_data = raw_gyro_data.rename(columns={"x": "gx", "y": "gy", "z": "gz"})


    combined_data = pd.DataFrame(columns = ["timestamp", "label", "ax", "ay", "az", "gx", "gy", "gz"])

    # get individual recordings and loop over them
    session_id = raw_accel_data['session_id'].unique()
    for i in range(len(session_id)):

        print(i)

        current_session_id = session_id[i]
        label = raw_accel_data['label'].unique()

        # print(current_session_id)
        # print(label[0])

        accel_data = raw_accel_data.loc[raw_accel_data['session_id'] == session_id[i]].reset_index(drop=True)
        gyro_data = raw_gyro_data.loc[raw_gyro_data['session_id'] == session_id[i]].reset_index(drop=True)

        print(accel_data.head(50))

        # print(accel_data.head(50))


        # convert timestamp columns to datetime datatype
        accel_data['timestamp'] = pd.to_datetime(accel_data['timestamp'], unit='ns', origin='unix').dt.floor('1ms')
        gyro_data['timestamp'] = pd.to_datetime(gyro_data['timestamp'], unit='ns', origin='unix').dt.floor('1ms')

        # set timestamp column as index
        accel_data.set_index('timestamp', inplace=True)
        gyro_data.set_index('timestamp', inplace=True)

        # print(accel_data['session_id'].unique())
        # print(gyro_data)

        print("********************")
        # resample both dataframes to a frequency of 100ms
        df_acc_resampled = accel_data.resample('100ms').mean()
        df_gyro_resampled = gyro_data.resample('100ms').mean()

        # merge the two resampled dataframes using the closest timestamp
        df_combined = pd.merge_asof(df_acc_resampled, df_gyro_resampled, on='timestamp')

        # interpolate any missing values from merging of both sensors using linear methods
        combined_timestamps = df_combined['timestamp']
        interpolated_values = df_combined[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].interpolate()


        data_label = pd.Series([label[0] for i in range(len(combined_timestamps))], name='label')

        result = pd.concat([combined_timestamps, data_label, interpolated_values], axis=1)

        print(result)


        combined_data = combined_data.concat(result)

        # print(result)
        # print(np.array(result))

        produce_graph_for_interpolated_data(result)

    print(combined_data)

        # result.to_csv('./combined_data.csv', index = False)



def produce_graph_for_interpolated_data(df):

    """ 
    Produce graphs that plot the accelerometer and gyroscope data for activities once sensors have been combined,
    frequency has been resampled to 100ms 
    
    """

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8,8))

    ax1.plot(df['timestamp'], df['ax'])
    ax2.plot(df['timestamp'], df['ay'])
    ax3.plot(df['timestamp'], df['az'])

    ax1.set_ylabel('Accelerometer X')
    ax2.set_ylabel('Accelerometer Y')
    ax3.set_ylabel('Accelerometer Z')
    fig.suptitle('Accelerometer Data')

    fig, (ax4, ax5, ax6) = plt.subplots(nrows=3, ncols=1, figsize=(8,8))

    ax4.plot(df['timestamp'], df['gx'])
    ax5.plot(df['timestamp'], df['gy'])
    ax6.plot(df['timestamp'], df['gz'])

    ax4.set_ylabel('Gyroscope X')
    ax5.set_ylabel('Gyroscope Y')
    ax6.set_ylabel('Gyroscope Z')
    fig.suptitle('Gyroscope Data')

    plt.show()

def produce_raw_data_graph(df, sensor):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8,8))

    ax1.plot(df['timestamp'], df['x'])
    ax2.plot(df['timestamp'], df['y'])
    ax3.plot(df['timestamp'], df['z'])

    ax1.set_ylabel('X axis')
    ax2.set_ylabel('Y axis')
    ax3.set_ylabel('Z axis')
    fig.suptitle(f'{sensor} Data')

    plt.show()

def main():
    
    CURRENT_PATH = os.getcwd()
    DATA_PATH = os.path.join(CURRENT_PATH, 'data')

    accel_data = pd.read_excel(os.path.join(DATA_PATH, 'accelerometer.xlsx'))
    gyro_data = pd.read_excel(os.path.join(DATA_PATH, 'gyroscope.xlsx'))

    # produce_raw_data_graph(accel_data, "Accelerometer")
    # produce_raw_data_graph(gyro_data, "Gyroscope")

    raw_data(accel_data, gyro_data)

if __name__ == "__main__":
    main()

