import pandas as pd
import plotly.express as px
from data_preprocess import load_dataset, preprocess_dataset
import plotly.graph_objects as go


def early_preprocess_dataset(df):
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df['date'] = pd.to_datetime(df['date'])
    return df


def stock_plot_history(df,title): 
    fig = go.Figure()
    columns = ['close', 'adj_close']
    for col in columns:
      fig.add_trace(go.Scatter(
        x = df['date'],
        y = df[col],
        mode = 'lines',
        name = col.capitalize() + ' Price')) 
    
    fig.update_layout(
    title = title,
    width=1500, 
    height=400,
    xaxis_title = 'Date',
    yaxis_title = 'Price')
    fig.write_image('../visualizations/' + title + '.webp')

if __name__ == '__main__':
   df_nvda,df_gspc = load_dataset()
   df_nvda = early_preprocess_dataset(df_nvda)
   df_gspc = early_preprocess_dataset(df_gspc)
   stock_plot_history(df_nvda,"Close Price and Adjusted Close Price of NVIDIA (NVDA) Stock")
   stock_plot_history(df_gspc,"Close Price and Adjusted Close Price of S&P 500 (GSPC) Index")
   