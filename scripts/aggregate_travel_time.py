from scripts.pre_processing import merge_file


def avg_travel_time():
    training_path = '../dataSets/training/'
    training_files = {
        'trajectories_file': 'trajectories(table 5)_training',
        'weather_file': 'weather (table 7)_training'
    }
    training_set = merge_file(training_path, **training_files)


def main():
    avg_travel_time()


if __name__ == '__main__':
    main()
