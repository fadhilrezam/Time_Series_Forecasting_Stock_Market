import logging
import pandas as pd
import matplotlib.pyplot as plt
import json
from prophet import Prophet
import mlflow
import mlflow.prophet
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

logging.basicConfig(
    filename = '../../logs/train_model.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s')

def load_preprocessed_dataset():
    try:
        train = pd.read_csv('../../data/processed/df_train.csv', parse_dates = ['ds'])
        test = pd.read_csv('../../data/processed/df_test.csv', parse_dates = ['ds'])
        logging.info('Data Loaded Successfully')
        return train, test
    except Exception as e:
        logging.info(f'Error Loading Data: {e}')

def train_model(df_train, df_test):
    try:
        json_path = '../../notebooks/best_params.json'
        with open(json_path, 'r') as f:
            params = json.load(f)
        prophet_model = Prophet(seasonality_mode='multiplicative').add_regressor('y_gspc')
        # prophet_model = Prophet(**params).add_regressor('y_gspc')
        future = df_test[['ds']].copy()
        future['y_gspc'] = df_test['y_gspc']
        prophet_model.fit(df_train)
        df_predicted = prophet_model.predict(future)[['ds','yhat']]
        signature = infer_signature(df_test, df_predicted)
        rmse = root_mean_squared_error(df_test['y'],df_predicted['yhat'])
        logging.info('Model Trained Successfully')
        return prophet_model, signature, df_predicted, rmse
    except Exception as e:
        logging.info(f'Error Training Model: {e}')

def plot_model ():
    plt.figure(figsize=(15,5))
    plt.plot(df_test['ds'], df_test['y'], label = 'Actual Price')
    plt.plot(df_predicted['ds'],df_predicted['yhat'], label = 'Prophet - Predicted Price')
    plt.title('Comparison of Actual and Predicted Price')
    plt.legend()
    plt.savefig('../visualizations/Comparison of Actual and Predicted Price.png')

def extract_params(model): #Extract prophet model parameter only with int, float, str and bool data type
    try:
        params_dict = vars(model)
        params = {key:value for key, value in params_dict.items() if isinstance(value, (int, float, str, bool))}
        logging.info(f'Prophet Params Successfully Extracted')
        return params
    except Exception as e:
        logging.info(f'Error Extracting Parameters: {e}')

if __name__ == '__main__':
    with mlflow.start_run() as run:
        df_train, df_test = load_preprocessed_dataset()
        prophet_model, signature,df_predicted, rmse_score = train_model(df_train, df_test)
        plot_model()
        params = extract_params(prophet_model)

        mlflow.prophet.log_model(prophet_model, artifact_path = 'prophet_model',signature = signature)
        mlflow.log_params(params)
        mlflow.log_metric('RMSE Score', rmse_score)

        logging.info('Process Completed')
