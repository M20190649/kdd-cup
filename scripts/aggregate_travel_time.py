import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from statsmodels.tsa.arima_model import ARIMA
import numpy as np

#############
# File path #
#############
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


def reshape_date(df):
    # Reshape the data
    data_shaped = []
    label_shaped = []
    starting_time = pd.DatetimeIndex(df['starting_time'])
    for name, group in df.groupby([starting_time.hour, starting_time.minute]):
        data_shaped.append(
            group.drop([
                'intersection_id', 'tollgate_id', 'starting_time', 'avg_travel_time', 'week', 'hour', 'minute'
            ], axis=1).values
        )
        label_shaped.append(group['avg_travel_time'].values)

    return np.asarray(data_shaped), np.asarray(label_shaped)


def rnn(data, label, batch_size):
    model = Sequential()
    model.add(
        LSTM(32, input_shape=(data.shape[0],), batch_size=batch_size, return_sequences=True, stateful=True)
    )
    model.add(LSTM(16, return_sequences=False, stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(data, label, batch_size=batch_size, epochs=1, shuffle=False)
    return model


def avg_travel_time():
    mape = []
    result_df = pd.DataFrame(
        columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']
    )
    # load training set
    for intersection, tollgates in route_dict.items():
        for tollgate in tollgates:
            # load train file
            training_file = '{}_{}_{}{}'.format('train', intersection, tollgate, file_suffix)
            training_set = pd.read_csv(file_path + training_file, parse_dates=['starting_time'])
            # load test file
            test_file = '{}_{}_{}{}'.format('test', intersection, tollgate, file_suffix)
            test_set = pd.read_csv(file_path + test_file, parse_dates=['starting_time'])

            # train trend model
            training_data, training_label = reshape_date(training_set)
            model = rnn(training_data, training_label, 1)

            result_df = result_df.append()

    result_df = result_df.reindex_axis(
        ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'], axis=1
    )
    result_df['tollgate_id'] = result_df['tollgate_id'].astype(int)
    # Prepare time window
    window_start = result_df['time_window'].astype(str)
    window_end = (result_df['time_window'] + pd.Timedelta(minutes=20)).astype(str)
    result_df['time_window'] = '[' + window_start + ',' + window_end + ')'
    result_df.to_csv('arima.csv', index=False)


def main():
    avg_travel_time()


if __name__ == '__main__':
    main()
