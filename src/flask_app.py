# from logger import logging
from exception import CustomException
import sys
import os

from pipeline.predict_pipeline import load_model_and_exog, predict_future
from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
app = Flask(__name__)


@app.route('/')
def hello_world():
    return ('hallo')

@app.route('/predict', methods = ['GET'])
def prediction():
    try:
        model_arima, df_exog = load_model_and_exog()
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        if not start_date_str:
            start_date_str = '2024-10-05'
        if not end_date_str:
            end_date_str = '2024-10-13'

        #Validate condition
        #1. start_date_str must be greater than 2024-10-04 
        #2. end_date_str must be greater than start_date_str

        if start_date_str <= '2024-10-04':
            return jsonify({'error': 'Start Date must be greater than 2024-10-04'})
        if end_date_str <= start_date_str:
            return jsonify({'error': 'End Date must be greater than Start Date'})
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        df_pred = predict_future(start_date,end_date, model_arima, df_exog)
        df_pred = df_pred[['close_pred_original_scale']]
        df_pred.index = df_pred.index.astype('str')

        print(df_pred)
    
        return jsonify(df_pred.to_dict(orient='index'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500 


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)