import os
import time
import pandas as pd


START_TIME = int(time.time())
LOG_FILENAME = 'log-{0}'.format(START_TIME)


def read_data(directory):
    train_features_path = os.path.join(directory, 'dengue_features_train.csv')
    train_labels_path = os.path.join(directory, 'dengue_labels_train.csv')
    test_features_path = os.path.join(directory, 'dengue_features_test.csv')
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



# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
