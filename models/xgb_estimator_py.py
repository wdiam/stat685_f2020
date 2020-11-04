import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np
import pandas as pd
import time
from models.utilities_py import split_by_time, data_info, prepare_x_y


default_param = {
    'nthread': [4], # when use hyperthread, xgboost may become slower
    'objective': ['reg:squarederror'],
    'learning_rate': [0.01],  # so called `eta` value
    'max_depth': [6, 8],
    'min_child_weight': [0],
    'silent': [1],
    'subsample': [0.75],
    'colsample_bytree': [0.7],
    'n_estimators': [750]
}


def cv_xgb_train(x_training, y_training, parameter_dict=None, x_testing=None, y_testing=None):
    """
    xgb estimator with GridSearchCV for parameter tuning
    :param x_training:
    :param y_training:
    :param parameter_dict:
    :param x_testing: optional
    :param y_testing: optional
    :return:
    """

    param = default_param if parameter_dict is None else parameter_dict

    x_g = GridSearchCV(xgb.XGBRegressor(),
                       param,
                       cv=10,
                       n_jobs=5,
                       verbose=100)

    if (x_testing is not None) and (y_testing is not None):
        # add eval_set
        mod = x_g.fit(x_training,
                      y_training,
                      eval_set=[(x_training, y_training), (x_testing, y_testing)],
                      verbose=100)
    else:
        mod = x_g.fit(x_training,
                      y_training,
                      verbose=100)
    mod = mod.best_estimator_
    return mod, x_g


def test_r2(y, yhat):
    """ calculate r2 """
    ybar = np.mean(y)
    ssres = np.sum((y-yhat)**2)
    sstot = np.sum((y - ybar)**2)
    return 1-(ssres/sstot)


def plot_validation(evals_result, metric='rmse', ax=None, title=None):
    """ generate validation plot """
    train_rmse = evals_result['validation_0'][metric]
    test_rmse = evals_result['validation_1'][metric]
    plot_dat = pd.DataFrame({'train_rmse': train_rmse,
                            'test_rmse': test_rmse})
    title = f"{metric} by iterations" if title is None else title
    plot_dat.plot(xlabel="iterations", ylabel=metric, ax=ax, title=title)


def _fit_model(train_df, test_df, track_eval=False, param=None):
    """
    fit a model and run testing
    :param train_df:
    :param test_df:
    :param track_eval:
    :param if not None, override parameter dict
    :return: result_dict, fit_mod.
    """
    res_tab = {}

    # data prepare
    train_x, train_y = prepare_x_y(train_df)
    test_x, test_y = prepare_x_y(test_df)

    # model training
    if track_eval:
        fit_mod, trained_grid = cv_xgb_train(x_training=train_x, y_training=train_y,
                                             x_testing=test_x, y_testing=test_y, parameter_dict=param)
    else:
        fit_mod, trained_grid = cv_xgb_train(x_training=train_x, y_training=train_y,
                                             x_testing=None, y_testing=None, parameter_dict=param)

    # calculate results
    res_tab['train_score'] = fit_mod.score(train_x, train_y)
    res_tab['mean_cv_score'] = trained_grid.cv_results_['mean_test_score'].mean()
    res_tab['test_score'] = fit_mod.score(test_x, test_y)
    # predict values
    y_pred = fit_mod.predict(test_x)
    predict_df = pd.DataFrame({"Predicted_Y_Test": y_pred, "Recorded_Y_Test": test_y})

    return res_tab, fit_mod, predict_df


def run_one_time_fit(data, split_date_str='01/11/2018', param=None):
    """
    forecasting using one fitted model
    :param data:
    :param split_date_str:
    :return:
    """
    train_df, test_df = split_by_time(data=data, date_str=split_date_str, hour=0)
    print(f"Training {data_info(train_df)}. Date size {train_df.shape}."
          f"\nTraining {data_info(test_df)}. Date size {test_df.shape}.")
    # start run
    start_time = time.time()
    sum_tab = {}  # collecting results

    # fit model
    res_tab, fit_mod, predict_df = _fit_model(train_df=train_df, test_df=test_df, track_eval=True, param=param)

    # update result
    sum_tab.update(res_tab)
    sum_tab['run_time'] = time.time() - start_time
    # prediction detail
    sum_tab['test_r2'] = test_r2(y=predict_df['Recorded_Y_Test'], yhat=predict_df['Predicted_Y_Test'])
    # return
    return sum_tab, predict_df, fit_mod


def _get_delta_hour_list(date_list, hour_list):
    """ get hour intervals """
    prev_date = None
    prev_hour = None
    delta_hour_list = []
    for date in date_list:
        for hour in hour_list:
            if prev_hour is None:
                prev_date = date
                prev_hour = hour
                continue
            # calculate hour interval from previous date
            days = int((date - prev_date) / np.timedelta64(1, 'D'))
            hour_delta = days * 24 + hour - prev_hour
            delta_hour_list.append(hour_delta)
            prev_date = date
            prev_hour = hour
    end_date = max(date_list)
    end_hour = 24
    days = int((end_date - prev_date) / np.timedelta64(1, 'D'))
    hour_delta = days * 24 + end_hour - prev_hour
    delta_hour_list.append(hour_delta)
    return delta_hour_list


def run_repeat_forecast(data, date_list, hour_list=None, param=None):
    """
    forecasting with repeated model fitting
    :param data:
    :param date_list: list of to repeat date object
    :param hour_list: list of to repeat daily hours (0 - 23)
    :param param: to override parameter dict
    :return:
    """
    # result collector
    start_time = time.time()
    sum_tab = {'train_score': [], 'mean_cv_score': [], 'test_score': []}
    predict_df = pd.DataFrame()

    # hour list check
    hour_list = [0] if hour_list is None else hour_list
    if max(hour_list) > 23 or min(hour_list) < 0:
        print("ERROR: hour out of range. check hour list")
        return None, None

    delta_hour_list = _get_delta_hour_list(date_list, hour_list)

    # start training for each day
    for day_idx, anchor_day in enumerate(date_list):
        # start training for each hour interval
        for hour_idx, anchor_hour in enumerate(hour_list):
            # get hours to next observation
            forecast_hours = delta_hour_list.pop(0)
            # generate train and test data
            train_df, test_df = split_by_time(data, '', date_obj=anchor_day, hour=anchor_hour)
            test_df = test_df.head(forecast_hours)
            print(f"Training {data_info(train_df)}. Date size {train_df.shape}."
                  f"\nTraining {data_info(test_df)}. Date size {test_df.shape}.")

            # model training
            res_tab, fit_mod, predict_df_sub = _fit_model(train_df=train_df, test_df=test_df,
                                                          track_eval=True, param=param)

            # update result
            sum_tab['train_score'].append(res_tab['train_score'])
            sum_tab['mean_cv_score'].append(res_tab['mean_cv_score'])
            sum_tab['test_score'].append(res_tab['test_score'])
            predict_df = predict_df.append(predict_df_sub)

    sum_tab['run_time'] = time.time() - start_time
    sum_tab['test_r2'] = test_r2(y=predict_df['Recorded_Y_Test'], yhat=predict_df['Predicted_Y_Test'])

    return sum_tab, predict_df


def printout_result(res_sum):
    df = pd.DataFrame(columns=['avg', 'std'])
    for k, v in res_sum.items():
        infostr = f"{k}: "
        if isinstance(v, float):
            infostr += f"{v: .4f}"
            sub_df = pd.DataFrame({'avg': v, 'std': np.nan}, index=[k])
        elif isinstance(v, list):
            infostr += f"avg {np.mean(v): .4f}; std {np.std(v): .4f}"
            sub_df = pd.DataFrame({'avg': np.mean(v), 'std': np.std(v)}, index=[k])
        else:
            infostr += "na"
            sub_df = pd.DataFrame({'avg': np.nan, 'std': np.nan}, index=[k])

        df = df.append(sub_df)

    return df


if __name__ == '__main__':
    from models.utilities_py import *
    dat = load_data()
    date_list = get_default_date_list(dat)
    hour_list = list(range(24))
    dat_hourly = dat.copy()
    # dat_hourly = add_past_hour_demand(dat_hourly, [24, 48, 24 * 7])
    print(dat_hourly.head(2))
    sum_t_hourly_2, p_df_hourly_2 = run_repeat_forecast(dat_hourly, date_list, None)

    # save result
    project_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '.'))
    p_df_hourly_2.to_csv(os.path.join(project_path, 'data/output_pred_daily_no_cor.csv'), index=True)
    import json

    with open(os.path.join(project_path, "data/output_daily_no_cor.json"), 'w') as fp:
        json.dump(sum_t_hourly_2, fp)

    sum_t_hourly_2_df = printout_result(sum_t_hourly_2)
    sum_t_hourly_2_df.to_csv(os.path.join(project_path, "data/output_daily_no_cor_sum.csv"))

    printout_result(sum_t_hourly_2)
