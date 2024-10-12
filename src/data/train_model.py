import logging
import pandas as pd
from prophet import Prophet
import mlflow
import mlflow.prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error


def load_preprocessed_dataset():
    try:
        train_scaled = pd.read_csv('../../data/processed/df_train_scaled.csv')
        test_scaled = pd.read_csv('../../data/processed/df_test_scaled.csv')
        print('data berhasil di load')
        return train_scaled, test_scaled
    except Exception as e:
        print(e)

def train_model(df_train_scaled, df_test_scaled):
    try:
        prophet_model = Prophet(seasonality_mode='multiplicative').add_regressor('y_gspc')
        future = df_test_scaled[['ds']].copy()
        future['y_gspc'] = df_test_scaled['y_gspc']
        prophet_model.fit(df_train_scaled)
        df_predicted = prophet_model.predict(future)[['ds','yhat']]
        rmse = root_mean_squared_error(df_predicted['yhat'], df_test_scaled['y'])

        with mlflow.start_run() as run:
            mlflow.prophet.log_model(prophet_model, 'prophet_model')
        return rmse
    except Exception as e:
        print(e)

if __name__ == '__main__':
    df_train_scaled, df_test_scaled = load_preprocessed_dataset()
    rmse_score = train_model(df_train_scaled, df_test_scaled)
    print('RMSE Score:', rmse_score)