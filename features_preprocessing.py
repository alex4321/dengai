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


def city_encode(df):
    df['city_sj'] = 1.0 * (df['city'] == 'sj')
    df['city_iq'] = 1.0 * (df['city'] == 'iq')
    return df


def ndvi_pows(df):
    columns = ['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw']
    for column in columns:
        df['{0}_2'.format(column)] = df[column] ** 2
        df['{0}_0_5'.format(column)] = df[column] ** 0.5
    return df


def choose_features(df):
    columns = ['ndvi_nw',
               'ndvi_sw',
               'reanalysis_air_temp_k',
               'reanalysis_avg_temp_k',
               'reanalysis_dew_point_temp_k',
               'reanalysis_min_air_temp_k',
               'reanalysis_precip_amt_kg_per_m2',
               'reanalysis_relative_humidity_percent',
               'reanalysis_sat_precip_amt_mm',
               'station_avg_temp_c',
               'station_diur_temp_rng_c',
               'station_max_temp_c',
               'station_min_temp_c',
               'station_precip_mm',
               'city_sj',
               'city_iq',

               'ndvi_se_2',
               'ndvi_sw_2',
               'ndvi_ne_2',
               'ndvi_nw_2',
               'ndvi_se_0_5',
               'ndvi_sw_0_5',
               'ndvi_ne_0_5',
               'ndvi_nw_0_5',

               'ds', # Special case to split final predictions
               'city',
               'year',
               'weekofyear'
               ]
    if 'y' in list(df.columns):
        columns.append('y')
    return df[columns]
    #return df
