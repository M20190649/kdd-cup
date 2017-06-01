import os
from datetime import datetime

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

#############
# File path #
#############
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

file_suffix = '.csv'
file_path = '../dataSets/travel_time2/'
sample_path = '../data/'

############
# Constant #
############
route_dict = {
    'A': [2, 3],
    'B': [1, 3],
    'C': [1, 3]
}

dropout_rates = {
    'A_2': 0.1, 'A_3': 0.4,
    'B_1': 0.4, 'B_3': 0.4,
    'C_1': 0.4, 'C_3': 0.4
}


def reshape_data(df):
    data = df.drop(['intersection_id', 'tollgate_id', 'starting_time', 'avg_travel_time'], axis=1).values
    label = df['avg_travel_time'].values
    # reshape
    data_shaped = np.reshape(data, (-1, 1, data.shape[1]))
    label_shaped = np.reshape(label, (-1, 1))

    return data_shaped, label_shaped


def rnn(data, label, batch_size):
    model = Sequential()

    model.add(LSTM(64, input_shape=data.shape[1:], batch_size=batch_size, return_sequences=True, stateful=True))

    model.add(LSTM(32, return_sequences=True, stateful=True))

    model.add(LSTM(16, return_sequences=False, stateful=True))

    model.add(Dense(label.shape[1]))

    model.compile(loss='mean_absolute_percentage_error', optimizer='RMSprop')

    model.fit(data, label, batch_size=batch_size, epochs=300, shuffle=False, verbose=0)

    return model


def rnn_train_on_batch(model, data, label):
    for i in range(len(data)):
        reshaped_data = np.reshape(data[i], (1, data[i].shape[0], data[i].shape[1]))
        reshaped_label = np.reshape(label[i], (1, label[i].shape[0]))
        model.train_on_batch(reshaped_data, reshaped_label)
    return model


def avg_travel_time():
    result_df = pd.DataFrame()
    for intersection, tollgates in route_dict.items():
        for tollgate in tollgates:
            print('Starting {}_{}'.format(intersection, tollgate))

            # Load train file
            training_file = '{}_{}_{}{}'.format('train', intersection, tollgate, file_suffix)
            training_set = pd.read_csv(file_path + training_file, parse_dates=['starting_time'])

            # Load test file
            test_file = '{}_{}_{}{}'.format('test', intersection, tollgate, file_suffix)
            test_set = pd.read_csv(file_path + test_file, parse_dates=['starting_time'])

            # Load submission file
            submission_file = '{}_{}_{}{}'.format('sub', intersection, tollgate, file_suffix)
            submission_set = pd.read_csv(file_path + submission_file, parse_dates=['starting_time'])

            # Train RNN model
            training_data, training_label = reshape_data(training_set)
            model = rnn(training_data, training_label, 1)

            # train on test batch first
            pre_label = []
            for every_sub_date in sub_date_list:
                for test_hour_min in test_hour_min_list:
                    test_start = every_sub_date + ' ' + test_hour_min['start']
                    test_end = every_sub_date + ' ' + test_hour_min['end']
                    test_time_range = pd.date_range(test_start, test_end, freq='20min')
                    batch = test_set[test_set['starting_time'].isin(test_time_range)]
                    reshaped_batch, reshaped_batch_label = reshape_data(batch)
                    model = rnn_train_on_batch(model, reshaped_batch, reshaped_batch_label)
                    sub_time_range = pd.date_range(
                        datetime.strptime(test_start, '%Y-%m-%d %H:%M:%S') + pd.Timedelta(hours=2),
                        datetime.strptime(test_end, '%Y-%m-%d %H:%M:%S') + pd.Timedelta(hours=2),
                        freq='20min')
                    sub_submission_data, sub_submission_label = reshape_data(
                        submission_set[submission_set['starting_time'].isin(sub_time_range)])
                    sub_predict = model.predict(sub_submission_data, batch_size=1).flatten()
                    pre_label.append(sub_predict)

            pre_label = np.array(pre_label).flatten()

            # Paper result DataFrame
            temp_df = pd.DataFrame({
                'intersection_id': intersection,
                'tollgate_id': tollgate,
                'starting_time': sub_date,
                'avg_travel_time': pre_label,
            })
            result_df = result_df.append(temp_df, ignore_index=True)

    ######################
    # Save result to csv #
    ######################
    # Prepare time window
    window_start = result_df['starting_time'].astype(str)
    window_end = (result_df['starting_time'] + pd.Timedelta(minutes=20)).astype(str)
    result_df['time_window'] = '[' + window_start + ',' + window_end + ')'
    # Save to the file
    result_df.drop(['starting_time'], axis=1).reindex_axis(
        ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'], axis=1
    ).to_csv('batch.csv', index=False)


def main():
    avg_travel_time()


if __name__ == '__main__':
    ############
    # Sub date #
    ############
    sub_date_list = pd.date_range(start='2016-10-25', end='2016-10-31', freq='D').format()
    sub_hour_min_list = [
        {'start': '08:00:00', 'end': '09:40:00'},
        {'start': '17:00:00', 'end': '18:40:00'}
    ]

    test_hour_min_list = [
        {'start': '06:00:00', 'end': '07:40:00'},
        {'start': '15:00:00', 'end': '16:40:00'}
    ]

    sub_date = []
    for date in sub_date_list:
        for hour_min in sub_hour_min_list:
            start = date + ' ' + hour_min['start']
            end = date + ' ' + hour_min['end']
            time_range = pd.date_range(start, end, freq='20min')
            sub_date.extend(time_range.values)

    ############
    # Run main #
    ############
    main()
