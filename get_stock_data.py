import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_stock_data(ticker_code, start_date, end_date):
    df_stock_data = yf.download(ticker_code, start = start_date, end = end_date)
    if df_stock_data.empty:
        print(f'Cannot find "{ticker_code}" stock data')
        return None
    else:
        df_stock_data.to_csv(f'data/raw/{ticker_code.lower()} stock prices data({start_date.date()} - {end_date.date()}).csv')
        return df_stock_data
    
def print_hello():
    return print('Hello, World!')



if __name__ == '__main__':
    ticker_code = input('Input the stock name: ')
    end_date = datetime.now()
    start_date = end_date - relativedelta(years = 3)

    # get_stock_data(ticker_code, start_date, end_date)
    get_stock_data(ticker_code, start_date, end_date)