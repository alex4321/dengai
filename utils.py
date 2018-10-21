import os
import pandas as pd


def read_data(directory):
    train_features_path = os.path.join(directory, 'dengue_features_train.csv')
    train_labels_path = os.path.join(directory, 'dengue_labels_train.csv')
    test_features_path = os.path.join(directory, 'dengue_features_train.csv')
    submission_path = os.path.join(directory, 'submission_format.csv')
    train_features = pd.read_csv(train_features_path)
    train_features['week_start_date'] = pd.to_datetime(train_features['week_start_date'])
    test = pd.read_csv(test_features_path)
    test['week_start_date'] = pd.to_datetime(test['week_start_date'])
    train = train_features.merge(train_labels_path,
                                 left_on=['city', 'year', 'weekofyear'],
                                 right_on=['city', 'year', 'weekofyear'])
    submission = pd.read_csv(submission_path)
    return train, test, submission
