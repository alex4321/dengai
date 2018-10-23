import numpy as np
import utils
from fbprophet.diagnostics import cross_validation
from fbprophet import Prophet
from tqdm import tqdm
import features_preprocessing as fp
from joblib import Parallel, delayed


def prophet_validation_error(prophet, postprocessors, *args, **kwargs):
    cv_data = cross_validation(prophet, *args, **kwargs)
    cv_data = utils.apply_processors(postprocessors, cv_data) \
        .rename(columns={'yhat': 'total_cases_predicted'}) \
        .rename(columns={'y': 'yhat'})
    cv_data = utils.apply_processors(postprocessors, cv_data) \
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


def _feature_validation(train, features, eliminated_feature, postprocessors, horizon):
    with utils.suppress_stdout_stderr():
        new_features = features[features != eliminated_feature]
        new_prophet = create_prophet(new_features)
        return prophet_validate(new_prophet, postprocessors, train, horizon=horizon)


if __name__ == '__main__':
    preprocessors = [fp.rename_columns_for_prophet,
                     fp.city_encode,
                     fp.fill_nan,
                     fp.choose_features,
                     fp.ysqrt]
    postprocessors = [fp.ysqrt_rev]
    max_depth = -1
    horizon = '{0} days'.format(2 * 365)
    max_proc = 4

    utils.log('Reading data')
    train, test, submission = utils.read_data('data')

    utils.log('Preprocessing')
    train = utils.apply_processors(preprocessors, train)
    test = utils.apply_processors(preprocessors, test)

    utils.log('Determine features')
    features = train.columns[
        (train.columns != 'year') &
        (train.columns != 'weekofyear') &
        (train.columns != 'city') &
        (train.columns != 'y') &
        (train.columns != 'ds')
    ]
    if max_depth == -1:
        max_depth = len(features)
    utils.log('Initial features: \n{0}'.format(train.dtypes[features]))

    utils.log('Initializing')
    with utils.suppress_stdout_stderr():
        prophet = create_prophet(features)
        error = prophet_validate(prophet, postprocessors, train, horizon=horizon)
    utils.log('Initial score: {0}'.format(error))

    for i in range(max_depth):
        utils.log('Iteration {0}'.format(i + 1))
        best_error = error
        best_eliminated_feature = None
        errors = Parallel(n_jobs=max_proc) (
            delayed(_feature_validation) (train, features, eliminated_feature, postprocessors, horizon)
            for eliminated_feature in tqdm(features)
        )
        for new_error, feature in zip(errors, features):
            if new_error < best_error:
                best_error = new_error
                best_eliminated_feature = feature
        if best_error < error:
            utils.log('Best eliminated feature on stage {0} is {1} with score {2}'.format(
                i + 1, best_eliminated_feature, best_error
            ))
            features = features[features != best_eliminated_feature]
            error = best_error
        else:
            utils.log('No improvement on stage {0}, stopping'.format(i + 1))
            break

    utils.log('Final features:\n{0}'.format(train.dtypes[features]))
    utils.log('Final error: {0}'.format(error))
