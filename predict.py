import utils
import features_preprocessing as fp
from fbprophet import Prophet
import pandas as pd


if __name__ == '__main__':
    preprocessors = [fp.rename_columns_for_prophet,
                     fp.fill_nan,
                     fp.choose_features,
                     fp.ysqrt]
    postprocessors = [fp.ysqrt_rev]

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

    utils.log('Training model')
    prophet = Prophet()
    for feature in features:
        prophet.add_regressor(feature)
    prophet.fit(train)

    utils.log('Making prediction')
    cities = test['city'].unique()
    city_predictions = []
    for city in cities:
        city_prediction = prophet.predict(test.loc[test['city'] == city])
        city_prediction['city'] = city
        city_predictions.append(city_prediction)
    prediction = pd.concat(city_predictions, sort=True)

    test_columns =
    pass