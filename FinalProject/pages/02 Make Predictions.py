import streamlit as st
import datetime

st.header("Please Pass the Required Inputs below: ")


predictform = st.form(key = "pform")
stock_selector = predictform.selectbox(
    "Select a Ticker",
    ["RELIANCE.NS"])
#ss_idx = all_stocks.index(stock_selector)
d = predictform.date_input(
    "Please select the Time Period for Predictions",
    value = (datetime.date(2001, 1, 1), datetime.date.today()),
    min_value = datetime.date(2001, 1, 1),
    max_value = datetime.date.today())

predictbutton = predictform.form_submit_button(label="Show Predictions")

if predictbutton:
    st.write()



