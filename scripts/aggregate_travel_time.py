
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from pre_processing import merge_file


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def avg_travel_time():
    training_path = '../dataSets/training/'
    # training_files = {
    #     'trajectories_file': 'trajectories(table 5)_training',
    #     'weather_file': 'weather (table 7)_training_update'
    # }
    # training_set = merge_file(training_path, **training_files)
    #
    test_path = '../dataSets/testing_phase1/'
    # test_files = {
    #     'trajectories_file': 'trajectories(table 5)_test1',
    #     'weather_file': 'weather (table 7)_test1'
    # }
    # test_set = merge_file(test_path, **test_files)
    #
    # # Extend training file
    # training_set = training_set.append(test_set)
    # training_set.to_csv(training_path + 'training.csv', index=False)

    # Training set
    training_set = pd.read_csv(training_path + 'training.csv', parse_dates=['starting_time'])
    x_training = training_set.drop(['starting_time', 'travel_time'], axis=1).reset_index()
    y_training = training_set['travel_time']

    # # Create test set
    # test_set = pd.read_csv('../data/' + 'submission_sample_travelTime.csv')
    # test_set['starting_time'] = test_set['time_window'].apply(
    #     lambda x: datetime.strptime(x.split(',')[0].split('[')[1], '%Y-%m-%d %H:%M:%S')
    # )
    # test_set = test_set.rename(columns={"avg_travel_time": "travel_time"})
    # test_set.to_csv(test_path + 'test.csv', index=False)

    test_files = {
        'trajectories_file': 'test',
        'weather_file': 'weather (table 7)_test1'
    }
    test_set = merge_file(test_path, **test_files)
    x_test = test_set.drop(['starting_time', 'travel_time'], axis=1).reset_index()

    # Encode intersection_id
    lb = preprocessing.LabelEncoder()
    x_training['intersection_id'] = lb.fit_transform(x_training['intersection_id'])
    x_test['intersection_id'] = lb.transform(x_test['intersection_id'])

    # Training model
    lr = LinearRegression()
    lr = lr.fit(x_training, y_training)

    test_set['avg_travel_time'] = lr.predict(x_test)
    test_set = test_set.round({'avg_travel_time': 2})
    # Create time window
    test_set['end'] = test_set['starting_time'] + pd.DateOffset(minutes=20)
    start_time = test_set['starting_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    end_time = test_set['end'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    test_set['time_window'] = '[' + start_time + ',' + end_time + ')'

    test_set = test_set[['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']]
    test_set.to_csv('LinearRegression.csv', index=False)


def main():
    avg_travel_time()


if __name__ == '__main__':
    main()
