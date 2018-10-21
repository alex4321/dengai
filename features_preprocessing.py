import numpy as np


def rename_columns_for_prophet(df):
    column_mapping = {
        'week_start_date': 'ds',
        'total_cases': 'y'
    }
    return df.rename(columns=column_mapping)


def fill_nan(df):
    return df.fillna(method='ffill')


def ysqrt(df):
    if 'y' in list(df.columns):
        df['y'] = df['y'].apply(np.abs).apply(np.sqrt) * df['y'].apply(np.sign)
    return df


def ysqrt_rev(df):
    if 'yhat' in list(df.columns):
        df['yhat'] = ((df['yhat'] ** 2) * df['yhat'].apply(np.sign)).apply(lambda value: max(0, value))
    return df


def choose_features(df):
    columns = ['ndvi_ne',
               'reanalysis_air_temp_k',
               'reanalysis_avg_temp_k',
               'reanalysis_dew_point_temp_k',
               'reanalysis_precip_amt_kg_per_m2',
               'reanalysis_relative_humidity_percent',
               'station_avg_temp_c',
               'station_diur_temp_rng_c',
               'ds', # Special case to split final predictions
               'city',
               'year',
               'weekofyear'
               ]
    if 'y' in list(df.columns):
        columns.append('y')
    return df[columns]
