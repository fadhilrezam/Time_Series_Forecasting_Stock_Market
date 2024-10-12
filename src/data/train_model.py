import logging
import pandas as pd
from prophet import Prophet
import mlflow
import mlflow.prophet
from mlflow.models.signature import infer_signature
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
        signature = infer_signature(df_test_scaled, df_predicted)
        rmse = root_mean_squared_error(df_predicted['yhat'], df_test_scaled['y'])
        return prophet_model, signature, rmse
    except Exception as e:
        print(e)

if __name__ == '__main__':
    df_train_scaled, df_test_scaled = load_preprocessed_dataset()
    prophet_model,signature, rmse_score = train_model(df_train_scaled, df_test_scaled)
    with mlflow.start_run() as run:
        mlflow.prophet.log_model(prophet_model, 'model',signature = signature)
    print('RMSE Score:', rmse_score)