import os
import time
import pandas as pd


START_TIME = int(time.time())
LOG_FILENAME = 'log-{0}'.format(START_TIME)


def read_data(directory):
    train_features_path = os.path.join(directory, 'dengue_features_train.csv')
    train_labels_path = os.path.join(directory, 'dengue_labels_train.csv')
    test_features_path = os.path.join(directory, 'dengue_features_train.csv')
    submission_path = os.path.join(directory, 'submission_format.csv')
    train_features = pd.read_csv(train_features_path)
    train_features['week_start_date'] = pd.to_datetime(train_features['week_start_date'])
    test = pd.read_csv(test_features_path)
    test['week_start_date'] = pd.to_datetime(test['week_start_date'])
    train_labels = pd.read_csv(train_labels_path)
    train = train_features.merge(train_labels,
                                 left_on=['city', 'year', 'weekofyear'],
                                 right_on=['city', 'year', 'weekofyear'])
    submission = pd.read_csv(submission_path)
    return train, test, submission


def log(message):
    message_str = '[{0}] {1}'.format(
        str(pd.to_datetime(time.time(), unit='s')),
        message
    )
    print(message_str)
    with open(LOG_FILENAME, 'a') as target:
        target.write(message_str + '\n')


def apply_processors(processors, df):
    for func in processors:
        df = func(df)
    return df
