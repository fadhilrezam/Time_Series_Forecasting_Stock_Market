from src.logger import logging
from src.exception import CustomException
import sys

from src.data_transform.data_ingestion import get_stock_data
from src.data_transform.data_preprocess import load_dataset, preprocess_dataset
from src.data_transform.train_model import train_model, save_model

from datetime import datetime
from dateutil.relativedelta import relativedelta

def run_pipeline(ticker_code, start_date, end_date):
    try:
        '''get_stock_data function
        -> Default :
        ticker_code = 'NVDA'
        start_date = 2019-10-05
        end_date = start_date + 5 years ahead (2024-10-05)'''
        get_stock_data(ticker_code, start_date,end_date)
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)

    try:
        '''load_dataset function
        -> Load dataset from local folder, where the path already defined in get_stock_data function'''
        df  = load_dataset()
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)

    try:
        '''preprocess_dataset function
        -> function that used to:
            1. Transform and normalize the dataframe such as transformation_log, differencing 
            2. Feature engineering which create new feautres such as lag and rolling window mean
            3. Split into train, test and prepare exogenous variables'''
        df, df_train, df_test, exog_train, exog_test, exog_future = preprocess_dataset(df)
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)
    
    try:
        '''train_model function
        -> function that used to train the model using ARIMA and store all the importan infortmation such as order, ar, ma parameters and rmse score in MLflow for tracking purposes'''
        df_pred_arima, model_arima, order, rmse_arima, rmse_arima_original_scale = train_model(df_train, df_test, exog_train, exog_test)
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)

    try:
        save_model(model_arima)
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)

if __name__ == '__main__':
    try:
        logging.info('Initializing Ticker Code and Date Range')
        ticker_code = input('Input the stock name (ex. TSLA, NVDA, ^GSPC, etc): ').strip().upper()
        start_date = input('Input start date (YYYY-MM-DD):')
        end_date = input('Input end date (YYYY-MM-DD):')
        if not ticker_code:
            ticker_code = 'NVDA'
        if not start_date:
            start_date = datetime(2019,10,5)
        if not end_date: 
            end_date = start_date + relativedelta(years = 5)
        run_pipeline(ticker_code, start_date, end_date)
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)
    





