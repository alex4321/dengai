import numpy as np
import utils
from fbprophet.diagnostics import cross_validation


def apply_processors(processors, df):
    for func in processors:
        df = func(df)
    return df


def validate_prophet(prophet, postprocessors, *args, **kwargs):
    cv_data = cross_validation(prophet, *args, **kwargs)
    cv_data = apply_processors(postprocessors, cv_data) \
        .rename(columns={'yhat': 'total_cases_predicted'}) \
        .rename(columns={'y': 'yhat'})
    cv_data = apply_processors(postprocessors, cv_data) \
        .rename(columns={'yhat': 'total_cases_true'})
    cv_data['error'] = (cv_data['total_cases_predicted'] - cv_data['total_cases_true']).apply(np.abs)
    fold_errors = cv_data.groupby('cutoff')[['error']].mean()
    return fold_errors, cv_data


# region preprocessors

def fill_nan(df):
    return df.fillna(method='ffill')

# endregion


if __name__ == '__main__':
    train, test, submission = utils.read_data('data')
    preprocessors = [fill_nan]
    postprocessors = []
    max_depth = -1

    column_mapping = {
        'week_start_date': 'ds',
        'total_cases': 'y'
    }
    train = train.rename(columns=column_mapping)
    test = test.rename(columns=column_mapping)
    train = apply_processors(preprocessors, train)
    test = apply_processors(preprocessors, test)

    features = train.columns

    cities = sorted(train['city'])
