import logging
import os
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

logging.basicConfig(
    filename = '../../logs/stock_data_retrieval.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s')


def get_stock_data(ticker_code, start_date, end_date):
    df_stock_data = yf.download(ticker_code, start = start_date, end = end_date)
    if df_stock_data.empty:
        logging.info('Error While Take Stock Data, Please Check the ticker code is right')
        ticker_code.replace("^","")
    else:
        df_stock_data.to_csv(f'../../data/raw/{ticker_code.replace("^","").lower()}_stock_prices_data({start_date.date()} - {end_date.date()}).csv')
        logging.info(f'{ticker_code} stock data saved successfully at ../../data/raw')

if __name__ == '__main__':
    ticker_code = input('Input the stock name: ').strip().upper()
    # end_date = datetime.now()
    end_date = input('Input last date (YYYY-MM-DD):')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    start_date = end_date - relativedelta(years = 5)

    get_stock_data(ticker_code, start_date, end_date)

    logging.info('Process Completed')