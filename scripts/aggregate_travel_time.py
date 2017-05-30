import os

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


def reshape_date(df):
    # Reshape the data
    data_shaped = []
    label_shaped = []
    starting_time = pd.DatetimeIndex(df['starting_time'])
    for name, group in df.groupby([starting_time.year, starting_time.month, starting_time.day]):
        data_shaped.append(
            group.drop([
                'intersection_id', 'tollgate_id', 'starting_time', 'avg_travel_time'
            ], axis=1).values
        )
        label_shaped.append(group['avg_travel_time'].values)

    return np.asarray(data_shaped), np.asarray(label_shaped)


def rnn(data, label, batch_size, dropout_rate):
    model = Sequential()

    model.add(LSTM(32, input_shape=data.shape[1:], batch_size=batch_size, return_sequences=True, stateful=True))
    model.add(Dropout(dropout_rate))

    model.add(LSTM(16, return_sequences=False, stateful=True))
    model.add(Dropout(dropout_rate))

    model.add(Dense(label.shape[1]))

    model.compile(loss='mean_absolute_percentage_error', optimizer='RMSprop')

    model.fit(data, label, batch_size=batch_size, epochs=200, shuffle=False, verbose=0)

    return model


def avg_travel_time():
    result_df = pd.DataFrame()
    mape_list = []
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
            training_data, training_label = reshape_date(training_set)
            dropout_rate = dropout_rates['{}_{}'.format(intersection, tollgate)]
            print('Drop out rate: {}'.format(dropout_rate))

            model = rnn(training_data, training_label, 1, dropout_rate)

            # Predict
            submission_data, submission_label = reshape_date(submission_set)
            pre_label = model.predict(submission_data, batch_size=1).flatten()

            # Cal mape
            test_date = test_set['starting_time']
            test_label = test_set['avg_travel_time']
            pre_label1 = pre_label[submission_set[submission_set['starting_time'].isin(test_date)].index.values]

            mape = np.mean(np.abs((test_label - pre_label1) / test_label)) * 100
            mape_list.append(mape)

            print("{}_{}: mape = {}\n".format(intersection, tollgate, mape))

            # Paper result DataFrame
            pre_label2 = pre_label[submission_set[submission_set['starting_time'].isin(sub_date)].index.values]
            temp_df = pd.DataFrame({
                'intersection_id': intersection,
                'tollgate_id': tollgate,
                'starting_time': sub_date,
                'avg_travel_time': pre_label2,
            })
            result_df = result_df.append(temp_df, ignore_index=True)

    print(mape_list)
    mean = np.mean(mape_list)
    print(mean)

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
    ).to_csv('dropOut.csv', index=False)


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
