from exception import CustomException
import sys

import streamlit as st
import datetime
from dateutil.relativedelta import relativedelta
import requests
import urllib.parse
import pandas as pd
import json
# import plotly.express as px

st.set_page_config(layout="wide")

min_value_start = datetime.date(2024,10,5)
min_value_end = min_value_start + relativedelta(days = 1)
start_date = st.date_input('Select your Start Date', value = datetime.date(2024,10,5), min_value = min_value_start)
end_date = st.date_input('Select your End Date', value = datetime.date(2024,10,13), min_value = min_value_end)

data = {
    'start_date':start_date.strftime('%Y-%m-%d'),
    'end_date':end_date.strftime('%Y-%m-%d')}

main_url = "http://localhost:8000/predict?"
url = main_url + urllib.parse.urlencode(data)


# @st.cache_data(show_spinner= False)
# def fetch_data(url):
#     r = requests.get(url, json = data)
#     try:
#         if r.status_code == 200:
#             return r.json()
#     except Exception as e:
#         st.error(CustomException(e,sys))

# json_response = fetch_data(url)
# # st.success("Done, Here are the results")

json_response = requests.get(url, json = data).json()
st.write(json_response)

if json_response:
    df = pd.DataFrame.from_dict(json_response, orient = 'index').rename(columns = {'close_pred_original_scale':'Predicted Close Price'})
    st.dataframe(df)
    st.line_chart(df, x_label = 'Date', y = 'Predicted Close Price')
    





