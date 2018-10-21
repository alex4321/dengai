import pandas as pd
import numpy as np
import utils
import time
from fbprophet.diagnostics import cross_validation
from fbprophet import Prophet
from tqdm import tqdm


START_TIME = int(time.time())
LOG_FILENAME = 'log-{0}'.format(START_TIME)


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


def prophet_validation_error(prophet, postprocessors, *args, **kwargs):
    cv_data = cross_validation(prophet, *args, **kwargs)
    cv_data = apply_processors(postprocessors, cv_data) \
        .rename(columns={'yhat': 'total_cases_predicted'}) \
        .rename(columns={'y': 'yhat'})
    cv_data = apply_processors(postprocessors, cv_data) \
        .rename(columns={'yhat': 'total_cases_true'})
    cv_data['error'] = (cv_data['total_cases_predicted'] - cv_data['total_cases_true']).apply(np.abs)
    fold_errors = cv_data.groupby('cutoff')[['error']].mean()
    return fold_errors, cv_data


def prophet_validate(prophet, postprocessors, train, *args, **kwargs):
    prophet.fit(train)
    errors, cv_data = prophet_validation_error(prophet, postprocessors, *args, **kwargs)
    return np.mean(errors['error'])


def create_prophet(features):
    prophet = Prophet()
    for feature in features:
        prophet.add_regressor(feature)
    return prophet


# region preprocessors

def fill_nan(df):
    return df.fillna(method='ffill')


def ysqrt(df):
    if 'y' in list(df.columns):
        df['y'] = df['y'].apply(np.abs).apply(np.sqrt) * df['y'].apply(np.sign)
    return df


def ysqrt_rev(df):
    if 'y' in list(df.columns):
        df['y'] = (df['y'] ** 2) * df['y'].apply(np.sign)
    return df


def choose_features(df):
    columns = ['ndvi_ne',
               'ndvi_se',
               'precipitation_amt_mm',
               'reanalysis_air_temp_k',
               'reanalysis_avg_temp_k',
               'reanalysis_dew_point_temp_k',
               'reanalysis_min_air_temp_k',
               'reanalysis_precip_amt_kg_per_m2',
               'reanalysis_relative_humidity_percent',
               'reanalysis_sat_precip_amt_mm',
               'station_avg_temp_c',
               'station_diur_temp_rng_c',
               'station_min_temp_c',
               'station_precip_mm',
               'ds',]
    if 'y' in list(df.columns):
        columns.append('y')
    return df[columns]

# endregion


if __name__ == '__main__':
    preprocessors = [fill_nan,
                     choose_features,
                     ysqrt]
    postprocessors = [ysqrt_rev]
    max_depth = -1
    horizon = '{0} days'.format(2 * 365)

    log('Reading data')
    train, test, submission = utils.read_data('data')

    log('Preprocessing')
    column_mapping = {
        'week_start_date': 'ds',
        'total_cases': 'y'
    }
    train = train.rename(columns=column_mapping)
    test = test.rename(columns=column_mapping)
    train = apply_processors(preprocessors, train)
    test = apply_processors(preprocessors, test)

    log('Determine features')
    features = train.columns[
        (train.columns != 'year') &
        (train.columns != 'weekofyear') &
        (train.columns != 'city') &
        (train.columns != 'y') &
        (train.columns != 'ds')
    ]
    if max_depth == -1:
        max_depth = len(features)
    log('Initial features: \n{0}'.format(train.dtypes[features]))

    log('Initializing')
    prophet = create_prophet(features)
    error = prophet_validate(prophet, postprocessors, train, horizon=horizon)
    log('Initial score: {0}'.format(error))

    for i in range(max_depth):
        log('Iteration {0}'.format(i + 1))
        best_error = error
        best_eliminated_feature = None
        for feature in tqdm(features,
                            desc='Choosing next feature for elimination'):
            new_features = features[features != feature]
            new_prophet = create_prophet(new_features)
            new_error = prophet_validate(new_prophet, postprocessors, train, horizon=horizon)
            if new_error < best_error:
                best_error = new_error
                best_eliminated_feature = feature
        if best_error < error:
            log('Best eliminated feature on stage {0} is {1} with score {2}'.format(
                i + 1, best_eliminated_feature, best_error
            ))
            features = features[features != best_eliminated_feature]
            error = best_error
        else:
            log('No improvement on stage {0}, stopping'.format(i + 1))
            break

    log('Final features:\n{0}'.format(train.dtypes[features]))
    log('Final error: {0}'.format(error))
