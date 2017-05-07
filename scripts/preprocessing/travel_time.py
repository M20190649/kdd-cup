from datetime import timedelta, datetime

import numpy as np
import pandas as pd

file_suffix = '.csv'
training_path = '../../dataSets/training/'
test_path = '../../dataSets/testing_phase1/'
output_path = '../../dataSets/travel_time/'


def interpolate_missing_value(data):
    """
    Interpolate missing starting_time 
    """
    starting_time = datetime(2016, 7, 19, 00, 00, 00)
    time = [int((i - starting_time).total_seconds()) for i in data['starting_time']]
    value = data['avg_travel_time'].values

    time_all = np.linspace(0, time[-1], (time[-1]) // 1200 + 1)
    value_all = np.interp(time_all, time, value)

    time_all = [starting_time + timedelta(seconds=i) for i in time_all]

    return pd.DataFrame(
        data={
            'intersection_id': np.unique(data['intersection_id'])[0],
            'tollgate_id': np.unique(data['tollgate_id'])[0],
            'starting_time': time_all,
            'avg_travel_time': value_all
        }
    )


def traj_weather(trajectories, weather, interpolate):
    data_set = trajectories[:]
    #####################
    # Load data_set #
    #####################
    # delete outliers
    data_set = data_set[data_set['travel_time'] < 1000]

    # calculate average travel time within time window 20min
    data_set = data_set.set_index(['starting_time'])
    data_set = data_set.groupby([
        'intersection_id', 'tollgate_id', pd.TimeGrouper('20Min')]
    )['travel_time'].mean().reset_index().rename(columns={'travel_time': 'avg_travel_time'})

    if interpolate:
        # insert missing time values
        repair_data = pd.DataFrame()
        for name, group in data_set.groupby(['intersection_id', 'tollgate_id']):
            repair_data = repair_data.append(interpolate_missing_value(group), ignore_index=True)

        data_set = repair_data.reindex_axis(
            ['intersection_id', 'tollgate_id', 'starting_time', 'avg_travel_time'], axis=1
        )

    # create date_hour in order to merge with weather
    date_hour = []
    # split date to week, hour, minute
    weeks = []
    hours = []
    minutes = []
    # set holiday
    moon_festival = pd.date_range(
        start=datetime(year=2016, month=9, day=15), end=datetime(year=2016, month=9, day=17)
    )
    national_holiday = pd.date_range(
        start=datetime(year=2016, month=10, day=1), end=datetime(year=2016, month=10, day=7)
    )
    holiday_range = moon_festival.append(national_holiday)
    holidays = []

    for start_time in data_set['starting_time']:
        # date_time belongs to 1:30h before or after the hour in weather
        time_delta = timedelta(hours=start_time.hour % 3, minutes=start_time.minute, seconds=start_time.second)
        weather_hour = start_time + timedelta(hours=3) if time_delta > timedelta(hours=1, minutes=30) else start_time
        date_hour.append(weather_hour - time_delta)

        weeks.append(datetime.strftime(start_time, format='%w'))
        hours.append(start_time.hour)
        minutes.append(start_time.minute)

        # check holiday
        is_holiday = 1 if start_time.date() in holiday_range else 0
        holidays.append(is_holiday)

    data_set['date_hour'] = date_hour
    data_set['week'] = weeks
    data_set['hour'] = hours
    data_set['minute'] = minutes
    data_set['holidays'] = holidays

    ################
    # Load weather #
    ################
    # Change wind_direction outlier
    outlier_index = weather.ix[weather['wind_direction'] == 999017].index.values
    for index in outlier_index:
        last = weather.loc[index - 1, 'wind_direction']
        nest = weather.loc[index + 1, 'wind_direction']
        weather.loc[index, 'wind_direction'] = (last + nest) / 2.0

    # Combine date and hour
    weather['date_hour'] = pd.to_datetime(
        weather['date'], format="%Y-%m-%d"
    ) + pd.to_timedelta(
        weather['hour'], unit="H"
    )
    # Delete unused column
    weather = weather.drop(['date', 'hour'], axis=1)

    #########
    # merge #
    #########
    data_set = pd.merge(data_set, weather, on='date_hour')

    data_set = data_set.drop('date_hour', axis=1)
    return data_set


def split_file(path, file_type, data):
    """
    Split file according to intersection_id and tollgate_id
    """
    for name, group in data.groupby(['intersection_id', 'tollgate_id']):
        file_name = '{}_{}_{}{}'.format(file_type, name[0], name[1], file_suffix)
        group.to_csv(path + file_name, index=False)


def main():
    #######################
    # load training files #
    #######################
    trajectories_file = training_path + 'trajectories(table 5)_training' + file_suffix
    trajectories = pd.read_csv(trajectories_file, parse_dates=['starting_time'])

    weather_file = training_path + 'weather (table 7)_training_update' + file_suffix
    weather = pd.read_csv(weather_file)

    training_set = traj_weather(trajectories, weather, True)

    ###################
    # load test files #
    ###################
    trajectories_file = test_path + 'trajectories(table 5)_test1' + file_suffix
    trajectories = pd.read_csv(trajectories_file, parse_dates=['starting_time'])

    weather_file = test_path + 'weather (table 7)_test1' + file_suffix
    weather = pd.read_csv(weather_file)

    test_set = traj_weather(trajectories, weather, False)

    ####################
    # Merge with route #
    ####################
    # load routes links file
    routes_links = pd.read_csv(training_path + 'route_link.csv')
    # merge them
    training_set = pd.merge(training_set, routes_links, on=['intersection_id', 'tollgate_id'], how='left')
    test_set = pd.merge(test_set, routes_links, on=['intersection_id', 'tollgate_id'], how='left')

    ##########
    # Export #
    ##########
    # export to separate files
    split_file(output_path, 'train', training_set)
    split_file(output_path, 'test', test_set)


if __name__ == '__main__':
    main()
