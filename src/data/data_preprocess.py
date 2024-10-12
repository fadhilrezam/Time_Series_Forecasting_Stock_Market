import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_dataset():
    try:
        nvda = pd.read_csv('../../data/raw/nvda stock prices data(2019-10-05 - 2024-10-05).csv')
        gspc = pd.read_csv('../../data/raw/gspc stock_prices_data(2019-10-05 - 2024-10-05).csv')
        return nvda, gspc
    except Exception as e:
        print(e)

def preprocess_dataset(dataframe):
    dataframe = dataframe[['Date', 'Close']]
    dataframe = dataframe.rename(columns = {'Date': 'ds', 'Close': 'y'})
    dataframe['ds'] = pd.to_datetime(dataframe['ds'])
    return dataframe

def merge_dataset(df_nvda, df_gspc):
    date_threshold = '2023-12-31'
    df = pd.merge(df_nvda,df_gspc[['ds','y']],on = 'ds', how = 'left', suffixes = ('','_gspc'))
    df_train = df[df['ds'] <= date_threshold]
    df_test = df[df['ds'] > date_threshold]
    return df_train, df_test

def normalize_dataset(df_train, df_test):
    scaler = MinMaxScaler()
    df_train_scaled = df_train[['ds']].copy()
    df_test_scaled = df_test[['ds']].copy()
    df_train_scaled[['y','y_gspc']] = scaler.fit_transform(df_train[['y','y_gspc']])
    df_test_scaled[['y','y_gspc']] = scaler.transform(df_test[['y','y_gspc']])
    return df_train_scaled, df_test_scaled

def store_train_test_data (df_train_scaled, df_test_scaled):
    df_train_scaled.to_csv('../../data/processed/df_train_scaled.csv', index = False)
    df_test_scaled.to_csv('../../data/processed/df_test_scaled.csv', index = False)

if __name__ == '__main__':
    df_nvda, df_gspc = load_dataset()
    df_nvda = preprocess_dataset(df_nvda)
    df_gspc = preprocess_dataset(df_gspc)
    df_train, df_test = merge_dataset(df_nvda, df_gspc)
    df_train_scaled, df_test_scaled = normalize_dataset(df_train, df_test)
    store_train_test_data(df_train_scaled, df_test_scaled)

    print ('Train Data (Tail)')
    print(df_train)
    print('Test Data (Head)')
    print(df_test)
    
