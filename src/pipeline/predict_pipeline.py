from logger import logging
from exception import CustomException
import sys
import os
import joblib
import pandas as pd
from datetime import datetime
import numpy as np


def load_model_and_exog():
    model_path = os.path.abspath(os.path.join(os.getcwd(),"..",'config'))
    exog_path = os.path.abspath(os.path.join(os.getcwd(),"..",'data','processed'))
    model_name = 'arima_model.pkl'
    exog_name = 'exog_future.csv'
    model_path = os.path.join(model_path, model_name)
    exog_path = os.path.join(exog_path, exog_name)
    try:
        logging.info('Load Model and Base Future Exogenous Variables')
        if os.path.exists(model_path) and os.path.exists(exog_path):
            model_arima = joblib.load(model_path)
            df_exog = pd.read_csv(exog_path, index_col = 0)
            return model_arima, df_exog
        else:
            logging.error('Model Path or Exog Path not found')
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)

def predict_future(start_date, end_date, model_arima, df_exog):
    try:
        logging.info('Prepare exogenous dataframe for future predictions')
        #Prepare exog dataframe based on date range
        all_dates = pd.date_range(start = start_date, end = end_date, freq = 'B')
        exog_future = pd.DataFrame({
            'lag_1': df_exog['lag_1'].values.tolist() * len(all_dates),
            'lag_2': df_exog['lag_2'].values.tolist()* len(all_dates),
            'lag_3': df_exog['lag_3'].values.tolist()* len(all_dates),
            'rolling_mean':  df_exog['rolling_mean'].values.tolist()* len(all_dates),
            'year': all_dates.year,
            'month': all_dates.month,
            'day': all_dates.day
        })

        logging.info('Prediction Initialization')
        y_pred = model_arima.get_forecast(steps = len(all_dates), exog = exog_future).predicted_mean.values

        df_pred = pd.DataFrame({
            'close_pred': y_pred}, index = all_dates)

        close_pred_reversed = df_pred['close_pred'].cumsum() + df_exog['close_log'].iloc[-1]
        close_pred_original_scale = round(np.exp(close_pred_reversed),3)
        df_pred['close_pred_original_scale']= close_pred_original_scale
        logging.info('Prediction Completed')

        return df_pred[['close_pred_original_scale']]
    
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e,sys)


if __name__ == '__main__':
    start_date = datetime(2024,10,5)
    end_date = datetime(2024,10,12)
    model_arima, exog_future = load_model_and_exog()
    df_pred = predict_future(start_date, end_date, model_arima, exog_future)