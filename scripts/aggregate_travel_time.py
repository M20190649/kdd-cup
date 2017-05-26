import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential

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
        LSTM(32, input_shape=data.shape[1:], batch_size=batch_size, return_sequences=True, stateful=True)
    )
    model.add(LSTM(16, return_sequences=False, stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_percentage_error', optimizer='RMSprop')
    model.fit(data, label, batch_size=batch_size, epochs=1, shuffle=False)
    return model


def avg_travel_time():
    mean = []
    result_df = pd.DataFrame(
        columns=['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']
    )
    # load training set
    for intersection, tollgates in route_dict.items():
        mape = []
        for tollgate in tollgates:
            # load train file
            training_file = '{}_{}_{}{}'.format('train', intersection, tollgate, file_suffix)
            training_set = pd.read_csv(file_path + training_file, parse_dates=['starting_time'])
            # load test file
            test_file = '{}_{}_{}{}'.format('test', intersection, tollgate, file_suffix)
            test_set = pd.read_csv(file_path + test_file, parse_dates=['starting_time'])

            # train RNN model
            # training_data, training_label = reshape_date(training_set)
            training_data = training_set.drop(
                ['intersection_id', 'tollgate_id', 'starting_time', 'avg_travel_time'], axis=1
            ).values
            # Reshape training data shape
            training_data = np.reshape(training_data, training_data.shape + (1,))
            training_label = training_set['avg_travel_time'].values
            model = rnn(training_data, training_label, 1)

            # Predict
            test_data = test_set.drop(
                ['intersection_id', 'tollgate_id', 'starting_time', 'avg_travel_time'], axis=1
            ).values
            test_data = np.reshape(test_data, test_data.shape + (1,))
            test_label = test_set['avg_travel_time'].values
            pre_label = model.predict(test_data, batch_size=1)

            mape.append(np.mean(np.abs((test_label - pre_label) / test_label)) * 100)

        mean.append(np.mean(mape))

    """
    result_df = result_df.reindex_axis(
        ['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time'], axis=1
    )
    result_df['tollgate_id'] = result_df['tollgate_id'].astype(int)
    # Prepare time window
    window_start = result_df['time_window'].astype(str)
    window_end = (result_df['time_window'] + pd.Timedelta(minutes=20)).astype(str)
    result_df['time_window'] = '[' + window_start + ',' + window_end + ')'
    result_df.to_csv('rnn.csv', index=False)
    """


def main():
    avg_travel_time()


if __name__ == '__main__':
    main()
