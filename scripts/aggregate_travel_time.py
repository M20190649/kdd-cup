import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np

#############
# File path #
#############
file_suffix = '.csv'
file_path = '../dataSets/travel_time/'
sample_path = '../data/'

############
# Constant #
############
route_dict = {
    'A': [2, 3],
    'B': [1, 3],
    'C': [1, 3]
}
date_list = pd.date_range(start='2016-10-18', end='2016-10-24', freq='D').format()
hour_min_list = [
    {'start': '08:00:00', 'end': '09:40:00'},
    {'start': '17:00:00', 'end': '18:40:00'}
]


def trend_model(df):
    result_df = pd.DataFrame(
        columns=['time_window', 'avg_travel_time']
    )
    training_set = df[['starting_time', 'avg_travel_time']].set_index('starting_time')
    # Training trend model
    for date in date_list:
        # Using trend_predict
        for hour_min in hour_min_list:
            start = date + ' ' + hour_min['start']
            end = date + ' ' + hour_min['end']
            train_series_interval = training_set.between_time(hour_min['start'], hour_min['end'])
            model = ARIMA(train_series_interval, order=(5, 1, 0))
            model_fit = model.fit(disp=0)

            time_range = pd.date_range(start, end, freq='20min')
            predictions = pd.Series(
                model_fit.forecast(steps=len(time_range))[0].tolist(), index=time_range
            )
            temp_df = pd.DataFrame({
                'time_window': predictions[start:end].index,
                'avg_travel_time': predictions[start:end].values
            }, columns=['time_window', 'avg_travel_time'])
            result_df = result_df.append(temp_df, ignore_index=True)

    return result_df


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
            temp_df = trend_model(training_set)
            temp_df['intersection_id'] = intersection
            temp_df['tollgate_id'] = tollgate

            result_df = result_df.append(temp_df)

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
