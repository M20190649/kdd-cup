from datetime import timedelta, datetime

import pandas as pd

file_suffix = '.csv'
training_path = '../../dataSets/training/'
test_path = '../../dataSets/testing_phase1/'
output_path = '../../dataSets/travel_volume/'


def vol_weather(volume, weather, predict):
    data_set = volume[:]
    #####################
    # Load data_set #
    #####################
    if not predict:
        # calculate the number of volume within time window 20min
        data_set = data_set.set_index(['time'])
        df_volume_per_window = pd.DataFrame(
            data_set.groupby(['tollgate_id', 'direction', pd.Grouper(freq='20Min')]).size())
        df_volume_per_window.columns = ['volume']

        df_volume_per_window = df_volume_per_window.reset_index()

        # calculate the number of each vehicle model with time-window (model0 - model7)
        for i in data_set['vehicle_model'].unique():
            temp_vehicle_model = data_set[data_set['vehicle_model'] == i][
                ['tollgate_id', 'direction', 'vehicle_model']].groupby(
                ['tollgate_id', 'direction', pd.Grouper(freq='20Min')]).size().reset_index()
            temp_vehicle_model.columns = ['tollgate_id', 'direction', 'time', 'vehicle_model_' + str(i)]
            df_volume_per_window = df_volume_per_window.merge(temp_vehicle_model,
                                                              on=['tollgate_id', 'direction', 'time'],
                                                              how='left')
            df_volume_per_window.fillna({'vehicle_model_' + str(i): 0}, inplace=True)

        # calculate the number of vehicles having etc and not having etc
        for i in data_set['has_etc'].unique():
            temp_has_etc = data_set[data_set['has_etc'] == i][['tollgate_id', 'direction', 'has_etc']].groupby(
                ['tollgate_id', 'direction', pd.Grouper(freq='20Min')]).size().reset_index()
            temp_has_etc.columns = ['tollgate_id', 'direction', 'time', 'has_etc_' + str(i)]
            df_volume_per_window = df_volume_per_window.merge(temp_has_etc, on=['tollgate_id', 'direction', 'time'],
                                                              how='left')
            df_volume_per_window.fillna({'has_etc_' + str(i): 0}, inplace=True)

        # calculate the number of vehicles per type (including unknown type)
        data_set['vehicle_type'].fillna('unknown', inplace=True)
        for i in data_set['vehicle_type'].unique():
            temp_vehicle_type = data_set[data_set['vehicle_type'] == i][
                ['tollgate_id', 'direction', 'vehicle_type']].groupby(
                ['tollgate_id', 'direction', pd.Grouper(freq='20Min')]).size().reset_index()
            temp_vehicle_type.columns = ['tollgate_id', 'direction', 'time', 'vehicle_type_' + str(i)]
            df_volume_per_window = df_volume_per_window.merge(temp_vehicle_type,
                                                              on=['tollgate_id', 'direction', 'time'],
                                                              how='left')
            df_volume_per_window.fillna({'vehicle_type_' + str(i): 0}, inplace=True)
    else:
        df_volume_per_window = volume

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

    training_set = vol_weather(volume, weather, False)

    ###################
    # load test files #
    ###################
    volume_file = test_path + 'volume(table 6)_test1' + file_suffix
    volume = pd.read_csv(volume_file, parse_dates=['time'])

    weather_file = test_path + 'weather (table 7)_test1' + file_suffix
    weather = pd.read_csv(weather_file)

    test_set = vol_weather(volume, weather, False)

    ##########
    # Export #
    ##########
    # export to separate files
    split_file(output_path, 'train', training_set)
    split_file(output_path, 'test', test_set)


if __name__ == '__main__':
    main()
