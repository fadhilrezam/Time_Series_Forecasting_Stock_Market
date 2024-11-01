from logger import logging
from exception import CustomException
import sys
import os

import pandas as pd
import numpy as np



    
def load_dataset():
    logging.info('Dataframe initialization')
    try:
        folder_path = os.path.abspath(os.path.join(os.getcwd(),"..","data","raw"))
        file_name = 'nvda_stock_prices.csv'
        file_path = os.path.join(folder_path,file_name)
        df = pd.read_csv(file_path)[['Date','Close']]
        logging.info('Dataframe successfully loaded')
        return df
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)

def preprocess_dataset(df):
    # Lower column name, replace whitespace and extract year, month and day from date column as new columns or features
    df.columns = df.columns.str.lower().str.replace(' ','_')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    #Transform dataframe with transformation log and differencing to stabilze the data range and remove any trends
    df['close_log'] = np.log(df['close'])
    df['close_log_diff'] = df['close_log'].diff()
    df.dropna(inplace = True)

    #Feature engineering with adding lag, rolling window, year, month and day column
    # df['lag_1'] = df['close_log_diff'].shift(1)
    # df['lag_2'] = df['close_log_diff'].shift(2)
    # df['lag_3'] = df['close_log_diff'].shift(3)
    for lag in range(1, 4):
        df[f'lag_{lag}'] = df['close_log_diff'].shift(lag)
    df['rolling_mean'] = df['close_log_diff'].rolling(window = 5).mean()
    df.dropna(inplace = True)

    # Split dataframe and to train test split with 80/20 distribution
    # Store other features as exogenous variables
    df.set_index('date', inplace = True)
    n_rows = int(len(df)*0.8)
    df_train = df.iloc[:n_rows]
    df_test = df.iloc[n_rows:]
    exog_train = df_train[['lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'year', 'month', 'day']]
    exog_test = df_test[['lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'year', 'month', 'day']]
    exog_future = df[['lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'year', 'month', 'day','close_log']].iloc[[-1]]

    return df_train, df_test, exog_train, exog_test, exog_future

def store_dataframe(df, df_train, df_test, exog_train, exog_test):
    logging.info('Prepare to store train and test data')
    try:
        folder_path = os.path.abspath(os.path.join(os.getcwd(),"..","data","processed")) 
        df.to_csv(os.path.join(folder_path, 'df.csv'))
        df_train.to_csv(os.path.join(folder_path, 'df_train.csv'))
        df_test.to_csv(os.path.join(folder_path, 'df_test.csv'))
        exog_train.to_csv(os.path.join(folder_path, 'exog_train.csv'))
        exog_test.to_csv(os.path.join(folder_path, 'exog_test.csv'))
        exog_future.to_csv(os.path.join(folder_path, 'exog_future.csv'))
        logging.info(f'All Dataframe Successfully stored in {folder_path}')
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    df = load_dataset()
    df_train, df_test, exog_train, exog_test, exog_future = preprocess_dataset(df)
    store_dataframe(df, df_train, df_test, exog_train, exog_test)
    # print([exog_future]*10)
    


