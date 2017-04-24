# import necessary modules
from datetime import timedelta, datetime

import pandas as pd

from sklearn import preprocessing

# set the data directory
file_suffix = '.csv'
path = '../dataSets/training/'

def avg_travel_time(**kwargs):
    trajectories_file = kwargs['trajectories_file'] + file_suffix
    weather_file = kwargs['weather_file'] + file_suffix

    file_type = kwargs['trajectories_file'].split('_')[1]

    #####################
    # Load trajectories #
    #####################
    trajectories = pd.read_csv(path + trajectories_file, parse_dates=['starting_time'])
    # Group in 20 minutes
    start_times = []
    for start_time in trajectories['starting_time']:
        start_times.append(start_time - timedelta(minutes=start_time.minute % 20, seconds=start_time.second, ))
    trajectories['starting_time'] = start_times
    # Cal average travel time for each route per time window
    trajectories = trajectories.groupby(
        ['intersection_id', 'tollgate_id', 'starting_time']
    )['travel_time'].mean().reset_index()
    trajectories = trajectories.round({'travel_time': 2})

    # Create date_hour in order to merge with weather
    date_hour = []
    for start_time in trajectories['starting_time']:
        # date_time belongs to 1:30h before or after the hour in weather
        time_delta = timedelta(hours=start_time.hour % 3, minutes=start_time.minute, seconds=start_time.second)
        if time_delta > timedelta(hours=1, minutes=30):
            start_time = start_time + timedelta(hours=3)
        date_hour.append(start_time - time_delta)
    trajectories['date_hour'] = date_hour

    ################
    # Load weather #
    ################
    weather = pd.read_csv(path + weather_file)
    # Combine date and hour
    weather['date_hour'] = pd.to_datetime(
        weather['date'], format="%Y-%m-%d"
    ) + pd.to_timedelta(
        weather['hour'], unit="H"
    )
    # Delete unused column
    weather = weather.drop(['date', 'hour'], axis=1)

    ##########
    # Merger #
    ##########
    training_set = pd.merge(trajectories, weather, on='date_hour')

    ######################
    # Convert to digital #
    ######################
    # Split date to week, hour, minute
    weeks = []
    hours = []
    minutes = []
    for start_time in training_set['starting_time']:
        weeks.append(datetime.strftime(start_time, format='%w'))
        hours.append(start_time.hour)
        minutes.append(start_time.minute)
    training_set['week'] = weeks
    training_set['hour'] = hours
    training_set['minute'] = minutes

    lb = preprocessing.LabelEncoder()
    training_set['intersection_id'] = lb.fit_transform(training_set['intersection_id'])

    training_set.to_csv(path + file_type + file_suffix, index=False)


def main():
    file_args = {
        'trajectories_file': 'trajectories(table 5)_training',
        'weather_file': 'weather (table 7)_training'
    }
    avg_travel_time(**file_args)


if __name__ == '__main__':
    main()
