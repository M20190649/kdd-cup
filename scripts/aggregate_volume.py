# -*- coding: utf-8 -*-
# !/usr/bin/env python
from datetime import datetime, timedelta

import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.preprocessing.traval_volume import vol_weather

# set the data directory
file_suffix = '.csv'
file_path = '../dataSets/travel_volume/'
sample_path = '../data/'


def avg_volume(in_file):
    tollgate_direction_dict = {
        1: [0, 1],
        2: [0],
        3: [0, 1]
    }
    # load weather
    weather_file = '../dataSets/testing_phase1/' + 'weather (table 7)_test1' + file_suffix
    weather = pd.read_csv(weather_file)

    # load sample
    sample_set = pd.read_csv(sample_path + 'submission_sample_volume.csv')

    result = []
    # load training set
    for tollgate, directions in tollgate_direction_dict.items():
        for direction in directions:
            # load train file
            training_file = '{}_{}_{}{}'.format('train', tollgate, direction, file_suffix)
            training_set = pd.read_csv(file_path + training_file, parse_dates=['time'])
            # load test file
            test_file = '{}_{}_{}{}'.format('test', tollgate, direction, file_suffix)
            test_set = pd.read_csv(file_path + test_file, parse_dates=['time'])

            cols = training_set.columns.values
            test_set = test_set[cols]

            # extend train file
            train = training_set.append(test_set)
            x_train = train.drop(['tollgate_id', 'direction', 'volume', 'time'], axis=1)
            y_train = train['volume']

            # load sample set
            sample_test = sample_set[
                (sample_set['tollgate_id'] == tollgate) & (sample_set['direction'] == direction)
                ]
            sample_test['time'] = sample_test['time_window'].apply(
                lambda x: datetime.strptime(x.split(',')[0].split('[')[1], '%Y-%m-%d %H:%M:%S')
            )
            sample_test['volume'] = 0
            sample_test.drop('time_window', axis=1, inplace=True)
            sample_test_dynamic_features = test_set.drop(
                ['tollgate_id', 'direction', 'volume', 'week', 'hour', 'minute', 'holidays', 'pressure',
                 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation'],
                axis=1)
            sample_test_dynamic_features['time'] = sample_test_dynamic_features['time'] + timedelta(hours=2)
            sample_test = pd.merge(sample_test, sample_test_dynamic_features, on='time')
            sample_test = vol_weather(sample_test, weather, True)
            x_test = sample_test.drop(['tollgate_id', 'direction', 'volume', 'time'], axis=1)

            # training model
            lr = LinearRegression()
            lr = lr.fit(x_train, y_train)
            # predict
            sample_test['volume'] = lr.predict(x_test)
            sample_test = sample_test.round({'volume': 2})

            result.append(sample_test)

    # Create time window
    result = pd.concat(result)
    result['end'] = result['time'] + pd.DateOffset(minutes=20)
    start_time = result['time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    end_time = result['end'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    result['time_window'] = '[' + start_time + ',' + end_time + ')'

    result = result[['tollgate_id', 'time_window', 'direction', 'volume']]
    result.to_csv('Volume_linearRegression_separate.csv', index=False)


def main():
    in_file = 'volume(table 6)_training'
    avg_volume(in_file)


if __name__ == '__main__':
    main()
