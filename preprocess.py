from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline
from sklearn import linear_model
import xgboost as xgb
from ultis import *

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def cast_log_outliers(to_file):
    df = pd.read_csv('raw/quaterfinal_gy_cmp_training_traveltime.txt', delimiter=';', dtype={'link_ID': object})
    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))

    df2 = pd.read_csv('raw/gy_contest_traveltime_training_data_second.txt', delimiter=';', dtype={'linkID': object})
    df2 = df2.rename(columns={"linkID": "link_ID"})
    df2['time_interval_begin'] = pd.to_datetime(df2['time_interval'].map(lambda x: x[1:20]))
    df2 = df2.loc[(df2['time_interval_begin'] >= pd.to_datetime('2017-03-01'))
                  & (df2['time_interval_begin'] <= pd.to_datetime('2017-03-31'))]

    df = pd.concat([df, df2])
    df = df.drop(['time_interval'], axis=1)
    df['travel_time'] = np.log1p(df['travel_time'])

    def quantile_clip(group):
        # group.plot()
        group[group < group.quantile(.05)] = group.quantile(.05)
        group[group > group.quantile(.95)] = group.quantile(.95)
        # group.plot()
        # plt.show()
        return group

    df['travel_time'] = df.groupby(['link_ID', 'date'])['travel_time'].transform(quantile_clip)
    df = df.loc[(df['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]

    print df.count()
    df.to_csv(to_file, header=True, index=None, sep=';', mode='w')


def imputation_prepare(file, to_file):
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})

    link_df = pd.read_csv('raw/gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})

    # date_range = pd.date_range("2016-07-01 00:00:00", "2016-07-31 23:58:00", freq='2min').append(
    #     pd.date_range("2017-04-01 00:00:00", "2017-07-31 23:58:00"))
    date_range = pd.date_range("2017-03-01 00:00:00", "2017-07-31 23:58:00", freq='2min')
    new_index = pd.MultiIndex.from_product([link_df['link_ID'].unique(), date_range],
                                           names=['link_ID', 'time_interval_begin'])

    new_df = pd.DataFrame(index=new_index).reset_index()
    df2 = pd.merge(new_df, df, on=['link_ID', 'time_interval_begin'], how='left')

    df2 = df2.loc[(df2['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]
    df2 = df2.loc[~((df2['time_interval_begin'].dt.year == 2017) & (df2['time_interval_begin'].dt.month == 7) & (
        df2['time_interval_begin'].dt.hour.isin([8, 15, 18])))]
    df2 = df2.loc[~((df2['time_interval_begin'].dt.year == 2017) & (df2['time_interval_begin'].dt.month == 3) & (
        df2['time_interval_begin'].dt.day == 31))]

    df2['date'] = df2['time_interval_begin'].dt.strftime('%Y-%m-%d')

    # check the missing values by date
    # df3.loc[(df3['travel_time'].isnull() == True)].groupby('date')['link_ID'].count().plot()
    # plt.show()

    print df2.count()

    df2.to_csv(to_file, header=True, index=None, sep=';', mode='w')


def imputation_with_model(file, to_file):
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'],
                     dtype={'link_ID': object})

    print df.describe()

    link_infos = pd.read_csv('raw/gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
    link_tops = pd.read_csv('raw/gy_contest_link_top.txt', delimiter=';', dtype={'link_ID': object})
    link_tops['in_links'] = link_tops['in_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops['out_links'] = link_tops['out_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops = link_tops.fillna(0)
    link_infos = pd.merge(link_infos, link_tops, on=['link_ID'], how='left')
    link_infos['area'] = link_infos['length'] * link_infos['width']
    link_infos['links_num'] = link_infos["in_links"].astype('str') + "," + link_infos["out_links"].astype('str')
    df = pd.merge(df, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'], how='left')

    df.loc[df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1
    df.loc[~df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

    df['hour'] = df['time_interval_begin'].dt.hour
    df['week_day'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df['month'] = df['time_interval_begin'].dt.month
    df['year'] = df['time_interval_begin'].dt.year

    df = pd.get_dummies(df, columns=['vacation', 'links_num', 'hour', 'week_day', 'month', 'year'])

    def mean_time(group):
        group['link_ID_en'] = group['travel_time'].mean()
        return group

    df = df.groupby('link_ID').apply(mean_time)
    sorted_link = np.sort(df['link_ID_en'].unique())
    df['link_ID_en'] = df['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))

    train_df = df.loc[~df['travel_time'].isnull()]
    test_df = df.loc[df['travel_time'].isnull()].copy()

    feature = df.columns.values.tolist()
    train_feature = [x for x in feature if
                     x not in ['link_ID', 'time_interval_begin', 'travel_time', 'date']]

    X = train_df[train_feature].values
    y = train_df['travel_time'].values

    print train_feature

    params = {
        'learning_rate': 0.2,
        'n_estimators': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'max_depth': 7,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'gamma': 0
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    eval_set = [(X_test, y_test)]
    regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                 booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'], random_state=0,
                                 max_depth=params['max_depth'], gamma=params['gamma'],
                                 min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric=mape_ln,
                  eval_set=eval_set)
    feature_vis(regressor, train_feature)

    test_df['prediction'] = regressor.predict(test_df[train_feature].values)
    df = pd.merge(df, test_df[['link_ID', 'time_interval_begin', 'prediction']], on=['link_ID', 'time_interval_begin'],
                  how='left')

    print df[['travel_time', 'prediction']].describe()
    df['imputation1'] = df['travel_time'].isnull()
    df['travel_time'] = df['travel_time'].fillna(value=df['prediction'])
    # print df.loc[df['travel_time'].isnull()].agg('count')['travel_time']
    df[['link_ID', 'date', 'time_interval_begin', 'travel_time', 'imputation1']].to_csv(to_file, header=True,
                                                                                        index=None, sep=';', mode='w')


def imputation_with_spline(file, to_file):
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})
    df['travel_time2'] = df['travel_time']

    def date_trend(group):
        tmp = group.groupby('date_hour').mean().reset_index()

        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        y = tmp['travel_time'].values
        nans, x = nan_helper(y)
        if group.link_ID.values[0] in ['3377906282328510514', '3377906283328510514', '4377906280784800514',
                                       '9377906281555510514']:
            tmp['date_trend'] = group['travel_time'].median()
        else:
            regr = linear_model.LinearRegression()
            regr.fit(x(~nans).reshape(-1, 1), y[~nans].reshape(-1, 1))
            tmp['date_trend'] = regr.predict(tmp.index.values.reshape(-1, 1)).ravel()
            # spl = UnivariateSpline(x(~nans), y[~nans])
            # tmp['date_trend'] = spl(tmp.index)
        group = pd.merge(group, tmp[['date_trend', 'date_hour']], on='date_hour', how='left')
        # plt.plot(tmp.index, tmp['date_trend'], 'o', tmp.index, tmp['travel_time'], 'ro')
        # plt.title(group.link_ID.values[0])
        # plt.show()
        return group

    df['date_hour'] = df.time_interval_begin.map(lambda x: x.strftime('%Y-%m-%d-%H'))
    df = df.groupby('link_ID').apply(date_trend)

    df = df.drop(['date_hour', 'link_ID'], axis=1)
    df = df.reset_index()
    df = df.drop('level_1', axis=1)
    df['travel_time'] = df['travel_time'] - df['date_trend']

    def minute_trend(group):
        tmp = group.groupby('hour_minute').mean().reset_index()
        spl = UnivariateSpline(tmp.index, tmp['travel_time'].values, s=0.5, k=3)
        tmp['minute_trend'] = spl(tmp.index)
        # plt.plot(tmp.index, spl(tmp.index), 'r', tmp.index, tmp['travel_time'], 'o')
        # plt.title(group.link_ID.values[0])
        # plt.show()
        # print group.link_ID.values[0]
        group = pd.merge(group, tmp[['minute_trend', 'hour_minute']], on='hour_minute', how='left')

        return group

    df['hour_minute'] = df.time_interval_begin.map(lambda x: x.strftime('%H-%M'))
    df = df.groupby('link_ID').apply(minute_trend)

    df = df.drop(['hour_minute', 'link_ID'], axis=1)
    df = df.reset_index()
    df = df.drop('level_1', axis=1)
    df['travel_time'] = df['travel_time'] - df['minute_trend']

    link_infos = pd.read_csv('raw/gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
    link_tops = pd.read_csv('raw/gy_contest_link_top.txt', delimiter=';', dtype={'link_ID': object})
    link_tops['in_links'] = link_tops['in_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops['out_links'] = link_tops['out_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops = link_tops.fillna(0)
    link_infos = pd.merge(link_infos, link_tops, on=['link_ID'], how='left')
    link_infos['links_num'] = link_infos["in_links"].astype('str') + "," + link_infos["out_links"].astype('str')
    link_infos['area'] = link_infos['length'] * link_infos['width']
    df = pd.merge(df, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'], how='left')

    df.loc[df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1

    df.loc[~df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

    df['minute'] = df['time_interval_begin'].dt.minute
    df['hour'] = df['time_interval_begin'].dt.hour
    df['day'] = df['time_interval_begin'].dt.day
    df['week_day'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df['month'] = df['time_interval_begin'].dt.month

    def mean_time(group):
        group['link_ID_en'] = group['travel_time'].mean()
        return group

    df = df.groupby('link_ID').apply(mean_time)
    sorted_link = np.sort(df['link_ID_en'].unique())
    df['link_ID_en'] = df['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))

    def std(group):
        group['travel_time_std'] = np.std(group['travel_time'])
        return group

    df = df.groupby('link_ID').apply(std)
    df['travel_time'] = df['travel_time'] / df['travel_time_std']


    params = {
        'learning_rate': 0.2,
        'n_estimators': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'max_depth': 10,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'gamma': 0
    }

    df = pd.get_dummies(df, columns=['links_num', 'width', 'minute', 'hour', 'week_day', 'day', 'month'])

    print df.head(20)

    feature = df.columns.values.tolist()
    train_feature = [x for x in feature if
                     x not in ['link_ID', 'time_interval_begin', 'travel_time', 'date', 'travel_time2', 'minute_trend',
                               'travel_time_std', 'date_trend']]

    train_df = df.loc[~df['travel_time'].isnull()]
    test_df = df.loc[df['travel_time'].isnull()].copy()

    print train_feature
    X = train_df[train_feature].values
    y = train_df['travel_time'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    eval_set = [(X_test, y_test)]
    regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                 booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'], random_state=0,
                                 max_depth=params['max_depth'], gamma=params['gamma'],
                                 min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_set=eval_set)

    test_df['prediction'] = regressor.predict(test_df[train_feature].values)

    df = pd.merge(df, test_df[['link_ID', 'time_interval_begin', 'prediction']], on=['link_ID', 'time_interval_begin'],
                  how='left')

    feature_vis(regressor,train_feature)

    df['imputation1'] = df['travel_time'].isnull()
    df['travel_time'] = df['travel_time'].fillna(value=df['prediction'])
    df['travel_time'] = (df['travel_time'] * np.array(df['travel_time_std']) + np.array(df['minute_trend'])
                         + np.array(df['date_trend']))

    print df[['travel_time', 'prediction', 'travel_time2']].describe()
    df[['link_ID', 'date', 'time_interval_begin', 'travel_time', 'imputation1']].to_csv(to_file, header=True,
                                                                                        index=None,
                                                                                        sep=';', mode='w')


def vis_imputation(df):
    def vis_minute_trend(group):
        group['travel_time'].plot()
        tmp = group.loc[group['imputation1'] == True]
        plt.scatter(tmp.index, tmp['travel_time'], c='r')
        plt.show()

    def vis_date_trend(group):
        group.groupby('date_hour').mean()['travel_time'].plot(figsize=(10, 15), title=group.link_ID.values[0])
        print group.link_ID.values[0]
        plt.show()

    # vis imputation for date_trend
    df['date_hour'] = df.time_interval_begin.map(lambda x: x.strftime('%Y-%m-%d-%H'))
    df.groupby('link_ID').apply(vis_date_trend)

    # vis imputation for minute_trend
    df['hour_minute'] = df.time_interval_begin.map(lambda x: x.strftime('%H-%M'))
    df.groupby(['link_ID', 'date']).apply(vis_minute_trend)


def create_lagging(df, df_original, i):
    df1 = df_original.copy()
    df1['time_interval_begin'] = df1['time_interval_begin'] + pd.DateOffset(minutes=i * 2)
    df1 = df1.rename(columns={'travel_time': 'lagging' + str(i)})
    df2 = pd.merge(df, df1[['link_ID', 'time_interval_begin', 'lagging' + str(i)]],
                   on=['link_ID', 'time_interval_begin'],
                   how='left')
    return df2


def create_feature(file, to_file, lagging=5):
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})

    # you can check imputation by uncomment the following:
    # vis_imputation(df)

    # lagging feature
    df1 = create_lagging(df, df, 1)
    for i in range(2, lagging + 1):
        df1 = create_lagging(df1, df, i)

    # length, width feature
    link_infos = pd.read_csv('raw/gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
    link_tops = pd.read_csv('raw/gy_contest_link_top.txt', delimiter=';', dtype={'link_ID': object})
    link_tops['in_links'] = link_tops['in_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops['out_links'] = link_tops['out_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops = link_tops.fillna(0)
    link_infos = pd.merge(link_infos, link_tops, on=['link_ID'], how='left')
    link_infos['links_num'] = link_infos["in_links"].astype('str') + "," + link_infos["out_links"].astype('str')
    link_infos['area'] = link_infos['length'] * link_infos['width']
    df2 = pd.merge(df1, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'], how='left')
    # df.boxplot(by=['width'], column='travel_time')
    # plt.show()
    # df.boxplot(by=['length'], column='travel_time')
    # plt.show()

    # links_num feature
    df2.loc[df2['links_num'].isin(['0.0,2.0', '2.0,0.0', '1.0,0.0']), 'links_num'] = 'other'
    # df.boxplot(by=['links_num'], column='travel_time')
    # plt.show()

    # vacation feature
    df2.loc[df2['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1
    df2.loc[~df2['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

    # minute_series for CV
    df2.loc[df2['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 6) * 60

    df2.loc[df2['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 13) * 60

    df2.loc[df2['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 16) * 60

    # day_of_week_en feature
    df2['day_of_week'] = df2['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df2.loc[df2['day_of_week'].isin([1, 2, 3]), 'day_of_week_en'] = 1
    df2.loc[df2['day_of_week'].isin([4, 5]), 'day_of_week_en'] = 2
    df2.loc[df2['day_of_week'].isin([6, 7]), 'day_of_week_en'] = 3

    # hour_en feature
    df2.loc[df['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'hour_en'] = 1
    df2.loc[df['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'hour_en'] = 2
    df2.loc[df['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'hour_en'] = 3

    # week_hour feature
    df2['week_hour'] = df2["day_of_week_en"].astype('str') + "," + df2["hour_en"].astype('str')

    # df2.boxplot(by=['week_hour'], column='travel_time')
    # plt.show()

    df2 = pd.get_dummies(df2, columns=['week_hour', 'links_num', 'width'])

    # ID Label Encode
    def mean_time(group):
        group['link_ID_en'] = group['travel_time'].mean()
        return group

    df2 = df2.groupby('link_ID').apply(mean_time)
    sorted_link = np.sort(df2['link_ID_en'].unique())
    df2['link_ID_en'] = df2['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))
    # df.boxplot(by=['link_ID_en'], column='travel_time')
    # plt.show()

    print df2.head(20)

    df2.to_csv(to_file, header=True, index=None, sep=';', mode='w')


if __name__ == '__main__':
    # cast_log_outliers('data/raw_data.txt')
    # imputation_prepare('data/raw_data.txt', 'data/pre_training.txt')
    # imputation_with_spline('data/pre_training.txt', 'data/com_training.txt')
    create_feature('data/com_training.txt', 'data/training.txt', lagging=5)
