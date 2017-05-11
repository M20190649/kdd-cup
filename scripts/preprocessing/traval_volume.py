from datetime import timedelta, datetime

import pandas as pd

file_suffix = '.csv'
training_path = '../../dataSets/training/'
test_path = '../../dataSets/testing_phase1/'
output_path = '../../dataSets/travel_volume/'


def vol_weather(volume, weather):
    data_set = volume[:]
    #####################
    # Load data_set #
    #####################
    # calculate the number of volume within time window 20min
    data_set = data_set.set_index(['time'])
    df_volume_per_window = pd.DataFrame(
        data_set.groupby(['tollgate_id', 'direction', pd.Grouper(freq='20Min')]).size())
    df_volume_per_window.columns = ['volume']
    df_volume_per_window = df_volume_per_window.reset_index()

    # create date_hour in order to merge with weather
    date_hour = []
    # split date to week, hour, minute
    weeks = []
    hours = []
    minutes = []
    # set holiday
    national_holiday = pd.date_range(
        start=datetime(year=2016, month=10, day=1), end=datetime(year=2016, month=10, day=7)
    )
    holiday_range = national_holiday
    holidays = []

    for time_window in df_volume_per_window['time']:
        # date_time belongs to 1:30h before or after the hour in weather
        time_delta = timedelta(hours=time_window.hour % 3, minutes=time_window.minute, seconds=time_window.second)
        weather_hour = time_window + timedelta(hours=3) if time_delta > timedelta(hours=1, minutes=30) else time_window
        date_hour.append(weather_hour - time_delta)

        weeks.append(datetime.strftime(time_window, format='%w'))
        hours.append(time_window.hour)
        minutes.append(time_window.minute)

        # check holiday
        is_holiday = 1 if time_window.date() in holiday_range else 0
        holidays.append(is_holiday)

    df_volume_per_window['date_hour'] = date_hour
    df_volume_per_window['week'] = weeks
    df_volume_per_window['hour'] = hours
    df_volume_per_window['minute'] = minutes
    df_volume_per_window['holidays'] = holidays

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
    df_volume_per_window = pd.merge(df_volume_per_window, weather, on='date_hour')

    df_volume_per_window = df_volume_per_window.drop('date_hour', axis=1)
    return df_volume_per_window


def split_file(path, file_type, data):
    """
    Split file according to tollgate_id and direction
    """
    for name, group in data.groupby(['tollgate_id', 'direction']):
        file_name = '{}_{}_{}{}'.format(file_type, name[0], name[1], file_suffix)
        group.to_csv(path + file_name, index=False)


def main():
    #######################
    # load training files #
    #######################
    volume_file = training_path + 'volume(table 6)_training' + file_suffix
    volume = pd.read_csv(volume_file, parse_dates=['time'])

    weather_file = training_path + 'weather (table 7)_training_update' + file_suffix
    weather = pd.read_csv(weather_file)

    training_set = vol_weather(volume, weather)

    ###################
    # load test files #
    ###################
    volume_file = test_path + 'volume(table 6)_test1' + file_suffix
    volume = pd.read_csv(volume_file, parse_dates=['time'])

    weather_file = test_path + 'weather (table 7)_test1' + file_suffix
    weather = pd.read_csv(weather_file)

    test_set = vol_weather(volume, weather)

    ##########
    # Export #
    ##########
    # export to separate files
    split_file(output_path, 'train', training_set)
    split_file(output_path, 'test', test_set)


if __name__ == '__main__':
    main()
