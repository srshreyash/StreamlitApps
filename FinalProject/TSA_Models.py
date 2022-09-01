import pickle
import numpy as np
import pandas as pd
import pmdarima as pm
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from datetime import datetime, timedelta
from yahoo_fin.stock_info import get_data
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, model_from_json

def auto_arima(stock_selector = "RELIANCE.NS", use_predefined_metrics = True, traintest_dates=["01/01/2015", "12/31/2021"],
               predict_dates = ["01/01/2022","07/18/2022"], predict_days = 185, lag = 1, model_path = None):
    
    traintest_date_from = traintest_dates[0]
    traintest_date_to = traintest_dates[1]
    predict_date_from = predict_dates[0]
    predict_date_to = predict_dates[1]
    predict_date_from = pd.to_datetime(traintest_date_to) + timedelta(days = 1)
    predict_date_to = pd.to_datetime(traintest_date_to) + timedelta(days = predict_days)
    train_df = get_data(stock_selector, start_date = traintest_date_from, end_date = traintest_date_to, index_as_date = True, interval="1d")
    predict_df = get_data(stock_selector, start_date = predict_date_from, end_date = predict_date_to, index_as_date = True, interval="1d")
    df = train_df.copy()
    idx = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(idx)
    df["close"] = df["close"].interpolate(method="linear")
    st.line_chart(df["close"])
    fig = plt.figure(figsize = (10, 5))
    lag_plot(df['close'], lag=lag)
    plt.title(f"{stock_selector} - Autocorrelation plot with lag = {lag}")
    st.pyplot(fig)
    if use_predefined_metrics:
      model = pickle.load(open(model_path, 'rb'))
    if not(use_predefined_metrics):
      model = pm.auto_arima(df.close, start_p=1, start_q=1,
                    test='adf',       # use adftest to find optimal 'd'
                    max_p=3, max_q=3, # maximum p and q
                    m=1,              # frequency of series
                    d=None,           # let model determine 'd'
                    seasonal=False,   # No Seasonality
                    start_P=0, 
                    D=0, 
                    trace=True,
                    error_action='ignore',  
                    suppress_warnings=True, 
                    stepwise=True)
      pickle.dump(model, open(model_path, 'wb')) #--- Save Model
    st.write(model.summary())
    fig = model.plot_diagnostics(figsize=(10,9))
    st.pyplot(fig)
    n_periods = pd.to_datetime(predict_date_to) - pd.to_datetime(predict_date_from)
    n_periods = n_periods.days
    n_periods = predict_days
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True) #Adding 1 because difference doesn't include last date
    df["date"] = df.index
    index_of_fc = np.arange(len(df.close), len(df.close)+n_periods)
    index_of_fc = pd.date_range(start=df["date"].iloc[-1], periods=n_periods+1, freq='d', closed='right')
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(df.close, label="Train Data - Actual")
    plt.plot(predict_df.close, label="Actual Values of Prediction")
    plt.plot(fc_series, color='darkgreen', label = "Prediction Values")
    plt.fill_between(lower_series.index, 
                      lower_series, 
                      upper_series, 
                      color='k', alpha=.15, label="Prediction Range")
    plt.legend()
    plt.title("Final Forecast")
    st.pyplot(fig)
    fig = plt.figure(figsize = (10, 5))
    plt.plot(predict_df.close, label="Actual Values of Prediction")
    plt.plot(fc_series, color='darkgreen', label = "Prediction Values")
    plt.fill_between(lower_series.index, 
                      lower_series, 
                      upper_series, 
                      color='k', alpha=.15, label="Prediction Range")
    plt.legend()
    st.pyplot(fig)

def LSTM_Model(stock_selector = "RELIANCE.NS" , use_predefined_metrics = True, traintest_dates=["01/01/2015", "12/31/2021"],
               predict_dates = ["01/01/2022","07/18/2022"], model_path = None, n=3):

    stock = stock_selector
    train_df = get_data(stock, start_date=traintest_dates[0], end_date=traintest_dates[1], index_as_date = True, interval="1d")
    predict_df = get_data(stock, start_date=predict_dates[0], end_date=predict_dates[1], index_as_date = True, interval="1d")
    df = train_df.copy()
    df["returns"] = df.close.pct_change()
    df["log_returns"] = np.log(1+ df["returns"])
    fig = plt.figure(figsize = (10, 5))
    plt.plot(df.log_returns, label = "Log Returns")
    plt.legend()
    st.pyplot(fig)
    df.dropna(inplace = True)
    X = df[["close","log_returns"]]
    scaler = MinMaxScaler(feature_range=(0,1)).fit(X)
    X_scaled = scaler.transform(X)
    y = [x[0] for x in X_scaled]
    split = int(len(X_scaled)*0.8)
    X_train = X_scaled[:split]
    X_test = X_scaled[split:len(X_scaled)]
    Y_train = y[:split]
    Y_test = y[split:len(y)]
    n = int(n)
    Xtrain=[]
    ytrain = []
    Xtest = []
    ytest = []
    for i in range(n, len(X_train)):
      Xtrain.append(X_train[i-n : i, :X_train.shape[1]])
      ytrain.append(Y_train[i])
    for i in range(n, len(X_test)):
      Xtest.append(X_test[i-n:i, :X_test.shape[1]])
      ytest.append(Y_test[i])
    Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))
    Xtest, ytest = (np.array(Xtest), np.array(ytest))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))
    if use_predefined_metrics:
        json_file = open(model_path+"/LSTM_model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_path+"/LSTM_model.h5")
        model.compile(loss='mean_squared_error', optimizer='adam')#, metrics=['accuracy'])
    if not(use_predefined_metrics):
        model = Sequential()
        ##        model.add(LSTM(4, input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
        ##        model.add(Dense(1))
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))

        model.add(Dense(units = 1))
        model.compile(loss="mean_squared_error", optimizer = "adam")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        model.fit(Xtrain, ytrain, epochs=100, validation_data=(Xtest, ytest), batch_size=16, verbose =1, callbacks = [early_stopping])
    model.summary(print_fn=lambda x: st.text(x))
    trainPredict = model.predict(Xtrain)
    testPredict = model.predict(Xtest)
    trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
    testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]
    trainPredict = scaler.inverse_transform(trainPredict)
    trainPredict = [x[0] for x in trainPredict]
    testPredict = scaler.inverse_transform(testPredict)
    testPredict = [x[0] for x in testPredict]
    trainScore = mean_squared_error(list(df["close"][n:split]), trainPredict, squared=False)
    st.write(f"Train Score: {trainScore}")
    testScore = mean_squared_error(list(df["close"][split+n:]), testPredict, squared=False)
    st.write(f"Test Score: {testScore}")
    dff = pd.DataFrame({"actual":list(df["close"][n:split]),"predicted":trainPredict})
    fig = plt.figure(figsize = (10, 5))
    plt.plot(dff["actual"], label="Actual")
    plt.plot(dff["predicted"], label="Predicted")
    plt.title("Model Performance on Train Data: ")
    plt.legend()
    st.pyplot(fig)
    dff = pd.DataFrame({"actual":list(df["close"][split+n:]),"predicted":testPredict})
    fig = plt.figure(figsize = (10, 5))
    plt.plot(dff["actual"], label = "Actual")
    plt.plot(dff["predicted"], label="Predicted")
    plt.title("Model Performance on Test Data: ")
    plt.legend()
    st.pyplot(fig)

def fb_prophet(stock_selector = "RELIANCE.NS" , use_predefined_metrics = True, traintest_dates=["01/01/2015", "12/31/2021"],
               predict_dates = ["01/01/2022","07/18/2022"], model_path = None):

    train_df = get_data(stock_selector, start_date = traintest_dates[0], end_date = traintest_dates[1], index_as_date = False, interval="1d")
    predict_df = get_data(stock_selector, start_date = predict_dates[0], end_date = predict_dates[1], index_as_date = False, interval="1d")
    df = train_df[["date","close"]]
    df.columns = ["ds","y"]
    df['ds']= pd.to_datetime(df['ds'])
    if use_predefined_metrics:
      model = pickle.load(open(model_path, 'rb'))
    if not(use_predefined_metrics):
        model = Prophet()
        mod = model.fit(df)
    df["ds"].min()
    pred = pd.DataFrame()
    pred["ds"] = df["ds"]
    predout = model.predict(pred)
    fig = model.plot(predout)
    plt.title("Training: ")
    plt.xlabel("Dates")
    plt.ylabel("Price")
    plt.legend(["Actual","Predicted","Confidence Interval"])
    st.pyplot(fig)
    pr = pd.DataFrame()
    pr["ds"] = list(pd.date_range("09/01/2022","09/30/2023"))
    prout = model.predict(pr)
    fig = model.plot(prout)
    plt.title("Prediction: ")
    plt.xlabel("Dates")
    plt.ylabel("Price")
    plt.legend(["Actual","Predicted","Confidence Interval"])
    st.pyplot(fig)

