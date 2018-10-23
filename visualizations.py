import matplotlib.pyplot as plt
import utils


def norm(arr):
    arr -= arr.min()
    arr /= arr.max() + 1e-10
    return arr


if __name__ == '__main__':
    utils.log('Reading data')
    train, test, submission = utils.read_data('data')
    train = train.loc[train['city'] == 'sj']
    del train['city']
    del train['year']
    del train['weekofyear']
    train = train.set_index('week_start_date')

    for column in train.columns:
        plt.figure(figsize=(15, 5))
        plt.plot(train.index, norm(train[column].fillna(method='ffill')))
        plt.plot(train.index, norm(train['total_cases']))
        plt.legend()
        plt.grid()
        plt.gcf().savefig('./visualizations/{0}-total_cases.png'.format(column))
        plt.close(plt.gcf())
        plt.figure(figsize=(15, 5))
        plt.plot(train.index, norm(train[column].fillna(method='ffill').rolling(window=4).mean()))
        plt.plot(train.index, norm(train['total_cases'].rolling(window=4).mean()))
        plt.legend()
        plt.grid()
        plt.gcf().savefig('./visualizations/rolling-{0}-total_cases.png'.format(column))
        plt.close(plt.gcf())
