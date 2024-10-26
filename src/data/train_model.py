from logger import logging
from exception import CustomException
import sys
import os

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error
import mlflow

import warnings
warnings.filterwarnings("ignore")


def load_preprocessed_dataset():
    logging.info('Train, Test and Exog dataframe initialization')
    try:
        folder_path = os.path.abspath(os.path.join(os.getcwd(),"..","data","processed"))
        df = pd.read_csv(os.path.join(folder_path, 'df.csv'), index_col=0)
        df_train = pd.read_csv(os.path.join(folder_path, 'df_train.csv'), index_col=0)
        df_test = pd.read_csv(os.path.join(folder_path, 'df_test.csv'), index_col=0)
        exog_train = pd.read_csv(os.path.join(folder_path, 'exog_train.csv'), index_col=0)
        exog_test =pd.read_csv(os.path.join(folder_path, 'exog_test.csv'), index_col=0)

        logging.info('Train and test dataframe loaded successfully')
        return df, df_train, df_test, exog_train, exog_test
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e,sys)

def train_model(df_train, df_test, exog_train, exog_test):
    logging.info('Training Model Initialization')
    try:
        order = (0,0,1)
        model_arima = ARIMA(df_train['close_log_diff'], order = order, exog = exog_train).fit() 
        y_pred_arima = model_arima.get_forecast(steps = len(df_test), exog = exog_test).predicted_mean.values

        df_pred_arima = pd.DataFrame({
            'date': df_test.index,
            'close_pred': y_pred_arima}, index = df_test.index)

        close_pred_reversed = df_pred_arima['close_pred'].cumsum() + df_train['close_log'].iloc[-1]
        close_pred_original_scale = np.exp(close_pred_reversed)
        df_pred_arima['close_pred_original_scale']= close_pred_original_scale

        rmse_arima = root_mean_squared_error(df_test['close_log_diff'], df_pred_arima['close_pred'])
        rmse_arima_original_scale = root_mean_squared_error(df_test['close'], df_pred_arima['close_pred_original_scale'])
        return df_pred_arima, model_arima, order, rmse_arima, rmse_arima_original_scale
    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e,sys)
if __name__ == '__main__':
    try:

        # Create a new MLflow Experiment
        mlflow.set_experiment("MLflow Stock Forecast with ARIMA")

        # Start an MLflow run
        with mlflow.start_run():
            logging.info("Loading preprocessed dataset")
            df, df_train, df_test, exog_train, exog_test = load_preprocessed_dataset()

            logging.info("Starting model training")
            df_pred_arima, model_arima, order, rmse_arima, rmse_arima_original_scale = train_model(df_train, df_test, exog_train, exog_test)

            mlflow.log_param("ARIMA_order", order)
            
            # Log AR and MA parameters
            for i, ar_param in enumerate(model_arima.arparams):
                mlflow.log_param(f"AR_param_{i+1}", ar_param)
            for i, ma_param in enumerate(model_arima.maparams):
                mlflow.log_param(f"MA_param_{i+1}", ma_param)
    
            # Log the loss metric
            mlflow.log_metric("RMSE Score Scaled", rmse_arima)
            mlflow.log_metric("RMSE Score Original Scale", rmse_arima_original_scale)


            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "ARIMA Model For NVIDIA stock price")

            logging.info("Training pipeline completed successfully")
    
    except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)
    