from datetime import datetime

import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.preprocessing.travel_time import traj_weather

file_suffix = '.csv'
file_path = '../dataSets/travel_time/'
sample_path = '../data/'


def avg_travel_time():
    route_dict = {
        'A': [2, 3],
        'B': [1, 3],
        'C': [1, 3]
    }
    # load weather
    weather_file = '../dataSets/testing_phase1/' + 'weather (table 7)_test1' + file_suffix
    weather = pd.read_csv(weather_file)

    # load routes links file
    routes_links_file = '../dataSets/training/' + 'route_link.csv'
    routes_links = pd.read_csv(routes_links_file)

    # load sample
    sample_set = pd.read_csv(sample_path + 'submission_sample_travelTime.csv')

    result = []
    # load training set
    for intersection, tollgates in route_dict.items():
        for tollgate in tollgates:
            # load train file
            training_file = '{}_{}_{}{}'.format('train', intersection, tollgate, file_suffix)
            training_set = pd.read_csv(file_path + training_file, parse_dates=['starting_time'])
            # load test file
            test_file = '{}_{}_{}{}'.format('test', intersection, tollgate, file_suffix)
            test_set = pd.read_csv(file_path + test_file, parse_dates=['starting_time'])

            # extend train file
            train = training_set.append(test_set)
            x_train = train.drop(['intersection_id', 'tollgate_id', 'avg_travel_time', 'starting_time'], axis=1)
            y_train = train['avg_travel_time']

            # load sample set
            sample_test = sample_set[
                (sample_set['intersection_id'] == intersection) & (sample_set['tollgate_id'] == tollgate)
                ]
            sample_test['starting_time'] = sample_test['time_window'].apply(
                lambda x: datetime.strptime(x.split(',')[0].split('[')[1], '%Y-%m-%d %H:%M:%S')
            )
            sample_test['travel_time'] = 0
            sample_test = traj_weather(sample_test, weather)
            sample_test = pd.merge(sample_test, routes_links, how='left')
            x_test = sample_test.drop(['intersection_id', 'tollgate_id', 'avg_travel_time', 'starting_time'], axis=1)

            # training model
            lr = LinearRegression()
            lr = lr.fit(x_train, y_train)
            # predict
            sample_test['avg_travel_time'] = lr.predict(x_test)
            sample_test = sample_test.round({'avg_travel_time': 2})

            result.append(sample_test)

    # Create time window
    result = pd.concat(result)
    result['end'] = result['starting_time'] + pd.DateOffset(minutes=20)
    start_time = result['starting_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    end_time = result['end'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    result['time_window'] = '[' + start_time + ',' + end_time + ')'

    result = result[['intersection_id', 'tollgate_id', 'time_window', 'avg_travel_time']]
    result.to_csv('LinearRegression_separate.csv', index=False)


def main():
    avg_travel_time()


if __name__ == '__main__':
    main()
