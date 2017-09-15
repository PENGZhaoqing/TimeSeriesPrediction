from preprocess import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from ultis import *

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def xgboost_submit(df, params):
    train_df = df.loc[df['time_interval_begin'] < pd.to_datetime('2017-07-01')]

    train_df = train_df.dropna()
    X = train_df[train_feature].values
    y = train_df['travel_time'].values

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
    joblib.dump(regressor, 'model/xgbr.pkl')
    print regressor
    submission(train_feature, regressor, df, 'submission/xgbr1.txt', 'submission/xgbr2.txt', 'submission/xgbr3.txt',
               'submission/xgbr4.txt')


def fit_evaluate(df, df_test, params):
    df = df.dropna()
    X = df[train_feature].values
    y = df['travel_time'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    df_test = df_test[valid_feature].values
    valid_data = bucket_data(df_test)

    eval_set = [(X_test, y_test)]
    regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                 booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'], random_state=0,
                                 max_depth=params['max_depth'], gamma=params['gamma'],
                                 min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    regressor.fit(X_train, y_train, verbose=False, early_stopping_rounds=10, eval_metric=mape_ln,
                  eval_set=eval_set)
    # feature_vis(regressor, train_feature)

    return regressor, cross_valid(regressor, valid_data,
                                  lagging=lagging), regressor.best_iteration, regressor.best_score


def train(df, params, best, vis=False):
    train1 = df.loc[df['time_interval_begin'] <= pd.to_datetime('2017-03-24')]
    train2 = df.loc[
        (df['time_interval_begin'] > pd.to_datetime('2017-03-24')) & (
            df['time_interval_begin'] <= pd.to_datetime('2017-04-18'))]
    train3 = df.loc[
        (df['time_interval_begin'] > pd.to_datetime('2017-04-18')) & (
            df['time_interval_begin'] <= pd.to_datetime('2017-05-12'))]
    train4 = df.loc[
        (df['time_interval_begin'] > pd.to_datetime('2017-05-12')) & (
            df['time_interval_begin'] <= pd.to_datetime('2017-06-06'))]
    train5 = df.loc[
        (df['time_interval_begin'] > pd.to_datetime('2017-06-06')) & (
            df['time_interval_begin'] <= pd.to_datetime('2017-06-30'))]

    regressor, loss1, best_iteration1, best_score1 = fit_evaluate(pd.concat([train1, train2, train3, train4]), train5,
                                                                  params)
    print (best_iteration1, best_score1, loss1)

    regressor, loss2, best_iteration2, best_score2 = fit_evaluate(pd.concat([train1, train2, train3, train5]), train4,
                                                                  params)
    print (best_iteration2, best_score2, loss2)

    regressor, loss3, best_iteration3, best_score3 = fit_evaluate(pd.concat([train1, train2, train4, train5]), train3,
                                                                  params)
    print (best_iteration3, best_score3, loss3)

    regressor, loss4, best_iteration4, best_score4 = fit_evaluate(pd.concat([train1, train3, train4, train5]), train2,
                                                                  params)
    print (best_iteration4, best_score4, loss4)

    regressor, loss5, best_iteration5, best_score5 = fit_evaluate(pd.concat([train2, train3, train4, train5]), train1,
                                                                  params)
    print (best_iteration5, best_score5, loss5)

    if vis:
        xgb.plot_tree(regressor, num_trees=5)
        results = regressor.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
        ax.legend()
        plt.ylabel('rmse Loss')
        plt.ylim((0.2, 0.3))
        plt.show()

    loss = [loss1, loss2, loss3, loss4, loss5]
    params['loss_std'] = np.std(loss)
    params['loss'] = str(loss)
    params['mean_loss'] = np.mean(loss)
    params['n_estimators'] = str([best_iteration1, best_iteration2, best_iteration3, best_iteration4, best_iteration5])
    params['best_score'] = str([best_score1, best_score2, best_score3, best_score4, best_score5])

    print str(params)
    if np.mean(loss) <= best:
        best = np.mean(loss)
        print "best with: " + str(params)
        feature_vis(regressor, train_feature)
    return best


lagging = 5

# cast_log_outliers('data/raw_data.txt')
# imputation_prepare('data/raw_data.txt', 'data/pre_training.txt')
imputation_with_spline('data/pre_training.txt', 'data/com_training.txt')
create_feature('data/com_training.txt', 'data/training.txt', lagging=lagging)
# add_links_lagging('data/training.txt', 'data/training1.txt')

df = pd.read_csv('data/training.txt', delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})
lagging_feature = ['lagging%01d' % e for e in range(lagging, 0, -1)]
base_feature = [x for x in df.columns.values.tolist() if x not in ['time_interval_begin', 'link_ID', 'link_ID_int',
                                                                   'date', 'travel_time', 'imputation1',
                                                                   'minute_series', 'area', 'hour_en', 'day_of_week']]
base_feature = [x for x in base_feature if x not in lagging_feature]
train_feature = list(base_feature)
train_feature.extend(lagging_feature)
valid_feature = list(base_feature)
valid_feature.extend(['minute_series', 'travel_time'])
print train_feature


# ----------------------------------------Train-------------------------------------------
params_grid = {
    'learning_rate': [0.05],
    'n_estimators': [100],
    'subsample': [0.6],
    'colsample_bytree': [0.6],
    'max_depth': [7],
    'min_child_weight': [1],
    'reg_alpha': [2],
    'gamma': [0]
}

grid = ParameterGrid(params_grid)
best = 1

for params in grid:
    best = train(df, params, best)

# ----------------------------------------submission-------------------------------------------
# submit_params = {
#     'learning_rate': 0.05,
#     'n_estimators': 100,
#     'subsample': 0.6,
#     'colsample_bytree': 0.6,
#     'max_depth': 7,
#     'min_child_weight': 1,
#     'reg_alpha': 2,
#     'gamma': 0
# }
#
# xgboost_submit(df, submit_params)
