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
#ss_idx = all_stocks.index(stock_selector)
# d = predictform.date_input(
#     "Please select the Time Period for Predictions",
#     value = (datetime.date(2001, 1, 1), datetime.date.today()),
#     min_value = datetime.date(2001, 1, 1),
#     max_value = datetime.date.today())

n_periods = predictform.number_input(label= "Please enter the number of days you would like to predict", min_value = 1, max_value = 365)
show_data = predictform.checkbox("Select to see the predictions in table")
predictbutton = predictform.form_submit_button(label="Show Predictions")

if predictbutton:
    predef = False
    # path = os.path.dirname(__file__)
    # st.header("ARIMA Modelling: ")
    # predict_days = 185
    # st.write("Please find the results of ARIMA modelling below: ")
    # model_path = r"C:\Users\hp\OneDrive\Documents\GitHub\StreamlitApps\FinalProject\SavedModels\ARIMA_model.pkl"
    # model = pickle.load(open(model_path, 'rb'))
    # st.write(model.summary())
    # n_periods = predict_days
    # fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    # df = pd.DataFrame(fc, columns=["predictions"])
    # from_date = datetime.datetime.today().date()+datetime.timedelta(days = 1)
    # to_date = from_date + datetime.timedelta(predict_days)
    # date_list = [from_date + datetime.timedelta(days=x) for x in range(n_periods)]
    # df["date"] = date_list
    # st.write(from_date)
    # st.write(to_date)
    # st.write(df)

    traintest_dates = ["01/01/2015", datetime.datetime.today().date()]
    #traintest_dates = [datetime.datetime.today().date(), datetime.datetime.today().date()+datetime.timedelta(predict_days)]
    #with st.expander("ARIMA"):
     #   auto_arima(stock_selector=stock_selector, use_predefined_metrics = False, predict_days = n_periods, traintest_dates = traintest_dates, predict_only = True, show_data = show_data)
    
    with st.expander("LSTM"):
        LSTM_Model(stock_selector=stock_selector, use_predefined_metrics = True, traintest_dates = traintest_dates, predict_only = True, show_data = show_data)

    #with st.expander("FBProphet"):
     #   fb_prophet(stock_selector = stock_selector, use_predefined_metrics = False, predict_days = n_periods, traintest_dates = traintest_dates, predict_only = True, show_data=show_data)
