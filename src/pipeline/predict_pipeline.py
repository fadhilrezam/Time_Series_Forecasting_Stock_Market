import os
from prophet.serialize import model_from_json
# from src.exception import CustomException
# from src.logger import logging
import sys
from datetime import datetime, timedelta
import pandas as pd
import holidays
from prophet import Prophet
import yfinance as yf


def load_model():
    json_path = os.path.join(os.path.dirname(__file__),'..','..','config')
    file_name = 'json_prophet_model.json'
    file_path = os.path.join(json_path, file_name)
    print(file_path)
    try:
        if os.path.exists(file_path):
            with open (file_path, 'r') as f: 
                model = model_from_json(f.read())
            return model
        else:
            print('file does not exist')
    except Exception as e:
        print(e)

def get_date():
    date_min = datetime(2024, 10, 6) 
    date_threshold = datetime (2024, 10, 5)
    date_max = date_min + timedelta(days=7)
    us_holidays = holidays.US()

    if date_min < date_threshold:
        print('Date not allowed, please raise the date over 5th of October, 2024')
    else:
        future_date = yf.download('^GSPC',start =date_min, end = date_max + timedelta(days=1)).reset_index().rename(columns = {'Date': 'ds', 'Close': 'y_gspc'})[['ds','y_gspc']]
        future_date = future_date[~(future_date.ds.isin(us_holidays) | future_date.ds.dt.dayofweek.isin([5,6]))] #exclude holidays date (include Saturday and Sunday)
        return future_date

def predict_future(model, future_date):
    df_predicted = model.predict(future_date)
    return df_predicted

def plot_future(model,df_predicted):
    fig = model.plot(df_predicted)
    fig.show()

if __name__ == '__main__':
    model = load_model()
    print(model)
    future_date = get_date()
    # df_predicted = model.predict(future_date)
    df_predicted = predict_future(model, future_date)
    print(df_predicted[['ds','yhat']])
    plot_future(model,df_predicted)
