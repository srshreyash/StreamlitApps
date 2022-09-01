import os
import time
import datetime
import streamlit as st
import yahoo_fin.stock_info as si
from TSA_Models import auto_arima, fb_prophet, LSTM_Model

def print_metrics():
    predef_train_from = "01/01/2015"
    predef_train_to = "12/31/2021"
    predef_predict_from = "07/01/2022"
    predef_predict_to = "07/18/2022"
    st.write(f"Training period : {predef_train_from} to {predef_train_to}")
    st.write(f"Prediction period : {predef_predict_from} to {predef_predict_to}")
    st.write("Please click on the Train button below to proceed further")

def create_form(all_stocks, formname = "Default", button_label = "Default", disabled = False):
    genericform = st.form(key = formname)
    stock_selector = genericform.selectbox(
        "Select a Ticker",
        all_stocks, disabled=disabled)
    ss_idx = all_stocks.index(stock_selector)
    d = genericform.date_input(
        "Please select the Time Period for Train/Test Data",
        value = (datetime.date(2001, 1, 1), datetime.date.today()),
        min_value = datetime.date(2001, 1, 1),
        max_value = datetime.date.today(), disabled = disabled)
    formbutton = genericform.form_submit_button(label=button_label)
    return formbutton

st.set_page_config(page_title="ML for Time Series Forecasting", page_icon="📈", layout="wide", initial_sidebar_state="collapsed", menu_items=None)
st.title("Application of Machine Learning Models for Time Series Analysis")

predef = st.checkbox("Use predefined Metrics", help = "Please select this to use the predefined metrics for model training and results")

if predef:
    print_metrics()
    path = os.path.dirname(__file__)
    train_all = st.button("Train All")
else:
    train_all = False
arima_tab, fbp_tab, LSTM_tab= st.tabs(["ARIMA","FBProphet","LSTM"])
all_stocks = si.tickers_nifty50()

with arima_tab:
    if predef:
        arimabutton = create_form(all_stocks=all_stocks,formname = "arima", button_label = "Train ARIMA", disabled=True)
        model = "..\SavedModels\ARIMA_model.pkl"
    if not(predef):
        arimabutton = create_form(all_stocks=all_stocks,formname = "arima", button_label = "Train ARIMA", disabled=False)
        model = None
    if arimabutton:
        st.write("SRS")
        auto_arima(model_path = model, use_predefined_metrics = predef)

with fbp_tab:
    if predef:
        fbpbutton = create_form(all_stocks=all_stocks,formname = "fbpform", button_label = "Train FBProphet", disabled=True)
        model = "..\SavedModels\FBProphet_model.pkl"
    if not(predef):
        fbpbutton = create_form(all_stocks=all_stocks,formname = "fbpform", button_label = "Train FBProphet", disabled=False)
        model = None
    if fbpbutton:
        fb_prophet(model_path = model, use_predefined_metrics = predef)

with LSTM_tab:
    if predef:
        lstmbutton = create_form(all_stocks=all_stocks,formname = "LSTMform", button_label = "Train LSTM", disabled = True)
        model = "..\SavedModels\LSTM_model.pkl"
    if not(predef):
        lstmbutton = create_form(all_stocks=all_stocks,formname = "LSTMform", button_label = "Train LSTM", disabled = False)
        model = None
    if lstmbutton:
        LSTM_Model(model_path = model, use_predefined_metrics = predef)
    
if train_all:
    st.header("ARIMA Modelling: ")
    st.write("Please find the results of ARIMA modelling below: ")
    start_time = time.time()
    model_path = path+'/SavedModels/ARIMA_model.pkl'
    model_path = "SavedModels/ARIMA_model.pkl"
    auto_arima(model_path = model_path, use_predefined_metrics = predef)
    st.write("Total Time taken : %s seconds" % (time.time() - start_time))
    st.header("FBProphet Modelling: ")
    st.write("Please find the results of FBProphet modelling below: ")
    start_time = time.time()
    model_path = path+'/SavedModels/FBProphet_model.pkl'
    fb_prophet(model_path = model_path, use_predefined_metrics = predef)
    st.write("Total Time taken : %s seconds" % (time.time() - start_time))
    st.header("LSTM Modelling: ")
    st.write("Please find the results of LSTM modelling below: ")
    start_time = time.time()
    model_path = path+'/SavedModels'
    LSTM_Model(model_path = model_path, use_predefined_metrics = predef)
    st.write("Total Time taken : %s seconds" % (time.time() - start_time))
