import streamlit as st
import pandas as pd
import os
import datetime
import time
import yahoo_fin.stock_info as si
from TSA_Models import auto_arima, fb_prophet, LSTM_Model
import pickle

st.header("Please Pass the Required Inputs below: ")
all_stocks = si.tickers_nifty50()
predictform = st.form(key = "pform")
stock_selector = predictform.selectbox(
    "Select a Ticker",
    all_stocks)

n_periods = predictform.number_input(label= "Please enter the number of days you would like to predict", min_value = 1, max_value = 365)
show_data = predictform.checkbox("Select to see the predictions in table")
predictbutton = predictform.form_submit_button(label="Show Predictions")

if predictbutton:
    predef = False
    traintest_dates = ["01/01/2015", datetime.datetime.today().date()]

    with st.expander("ARIMA"):
       auto_arima(stock_selector=stock_selector, use_predefined_metrics = False, predict_days = n_periods, traintest_dates = traintest_dates, predict_only = True, show_data = show_data)
    
    with st.expander("LSTM"):
        LSTM_Model(stock_selector=stock_selector, predict_days = n_periods, use_predefined_metrics = False, traintest_dates = traintest_dates, predict_only = True, show_data = show_data)

    with st.expander("FBProphet"):
       fb_prophet(stock_selector = stock_selector, use_predefined_metrics = False, predict_days = n_periods, traintest_dates = traintest_dates, predict_only = True, show_data=show_data)
