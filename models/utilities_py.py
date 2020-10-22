"""
common functions
"""
import os
import pandas as pd
import numpy as np
from scipy.stats.contingency import margins


def load_data():
    """
    read data and set dtypes
    :return: pd.DataFrame
    """
    fn = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), './data/SeoulBikeData.csv'))
    colNames = ["Date", "RentedBikeCount", "Hour", "Temp", "Humidity", "WindSpeed", "Visibility",
                "DewPointTemp", "SolarRadiation", "Rainfall", "Snowfall", "Seasons", "Holiday", "FunctionalDay"]
    dat = pd.read_csv(fn, encoding="ISO-8859-1")
    dat.columns = colNames
    dat = dat.astype(
        {"Date": str,
         "RentedBikeCount": int,
         "Hour": float,
         "Temp": float,
         "Humidity": float,
         "WindSpeed": float,
         "Visibility": float,
         "DewPointTemp": float,
         "SolarRadiation": float,
         "Rainfall": float,
         "Snowfall": float,
         "Seasons": str,
         "Holiday": str,
         "FunctionalDay": str})

    # convert qualitative variable to quantitative
    dat = pd.get_dummies(dat, columns=["Seasons"])
    del dat['Seasons_Winter']
    dat = pd.get_dummies(dat, columns=["Holiday"])
    del dat['Holiday_Holiday']
    dat = pd.get_dummies(dat, columns=["FunctionalDay"])
    del dat['FunctionalDay_Yes']

    # converting date to datetime obj
    dat['Date'] = pd.to_datetime(dat['Date'], format="%d/%m/%Y")

    return dat


def split_by_time(data, date_str, date_obj=None, hour=None):
    """
    param date_str: string in format "%d/%m/%Y"
    date_obj: numpy.datetime64
    param hour: integer
    param dat: pd.DataFrame
    return: pd.DataFrame(before), pd.DataFrame(after)
    """
    anchor_date = pd.to_datetime(date_str, format="%d/%m/%Y") if date_obj is None else date_obj
    if hour is None:
        prev = data[data['Date'] < anchor_date]
        after = data[data['Date'] >= anchor_date]
        return prev, after
    # if hour is defined
    prev = data[(data['Date'] < anchor_date) |
               ((data['Date']== anchor_date)&(data['Hour'] < hour))]
    after = data[(data['Date'] > anchor_date) |
                ((data['Date']== anchor_date) & (data['Hour'] >= hour))]
    return prev, after


def data_info(df):
    """
    get data set date/hour from to information
    :param df:
    :return: str
    """
    min_date = df['Date'].dt.date.min()
    max_date = df['Date'].dt.date.max()
    min_hour = df[df['Date'] == df['Date'].min()]['Hour'].min()
    max_hour = df[df['Date'] == df['Date'].max()]['Hour'].max()
    return f"from {str(min_date)} hour {min_hour} to {str(max_date)} hour {max_hour}."


def prepare_x_y(df):
    """
    split data set into X and Y
    :param df:
    :return:
    """
    X_df = df.iloc[:, 2:df.shape[1]]
    Y_df = df.iloc[:,1]
    return X_df, Y_df


def add_lag(lag, dat):
    """
    add past Y information
    :param lag: by hour
    :param dat:
    :return: list
    """
    lag_demand = list(dat.loc[0:len(dat)-lag-1, 'RentedBikeCount'])
    fill_lag = list([np.nan] * lag)
    fill_lag.extend(lag_demand)
    return fill_lag


def add_past_hour_demand(dat, past_hour_list):
    """
    add x hour ago demand
    :param dat: pd.DataFrame
    :param past_hour_list: list of hour lag
    :return: pd.DataFrame with additional columns
    """
    for lag_hour in past_hour_list:
        dat[f"lag{lag_hour}"] = add_lag(lag_hour, dat)
    dat = dat.dropna()
    return dat


def get_default_date_list(data):
    date_list = sorted(data[data['Date'].dt.month == 11]['Date'].unique())
    return date_list


def compare_table(res_dict):
    res_com_df = None
    for k, res_sum in res_dict.items():
        if res_com_df is None:
            res_com_df = pd.DataFrame(index=list(res_sum.keys()))
        res_com_df[k] = [np.mean(v) if isinstance(v, list) else v for v in res_sum.values()]
    return res_com_df


def cal_std_resid(observed, expected):
    # Calculate Standardized Residuals
    res = expected - observed
    re_sd = np.std(res)
    return res / re_sd