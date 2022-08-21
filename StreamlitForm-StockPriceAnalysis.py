import streamlit as st
import numpy as np
import pandas as pd
#import cufflinks
from yahoo_fin.stock_info import get_data
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import plotly.graph_objects as go
from PIL import Image
import base64
import io
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

st.set_page_config(page_title="Ellaam Maayai", page_icon="🤘", layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title(f"SIP Date Analysis")
st.markdown("This app has been designed to answer the question 'Is there any particular date/week for SIP which yields higher returns than others?'")

def get_stock_data(ticker, start_date, end_date):
    stock_data= get_data(ticker, start_date=start_date.strftime("%m/%d/%Y"), end_date=end_date.strftime("%m/%d/%Y"), index_as_date = False, interval='1d')
    return stock_data
    
all_stocks = si.tickers_nifty50()

sidebar = st.sidebar
sidebar.header("List of 30 stocks trading in the National Stock Exchange of India is fetched from yahoo and displayed below. Please select one.")


with sidebar.expander("Open for having a peek into the data"):
    form = st.form(key='my_form')
    stock_selector = form.selectbox(
        "Select a Ticker",
        all_stocks)
    ss_idx = all_stocks.index(stock_selector)
    d = form.date_input(
        "Please select the Time Period for analysis",
        value = (datetime.date(2001, 1, 1), datetime.date.today()),
        min_value = datetime.date(2001, 1, 1),
        max_value = datetime.date.today(), key = "daterange")
    trend_level = form.selectbox("Trend Level", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                               help="This will group and display data based on your preference")
    chart_selector = form.selectbox("Chart Type: ", ('Line','Bar','Candle Stick'),
                                    help="Select your favourite chart type for visualizing data. My Personal Fav: Candlestick ❤️‍🔥")
    show_data = form.checkbox("Show Data",key="showdata")
    data_peek = form.form_submit_button(label='Peek into the data')

with sidebar.expander("Expand for Finding Minimums Every Month"):
    minform = st.form(key='min_form')
    stock_selector = minform.selectbox(
        "Select a Ticker",
        all_stocks, index = ss_idx)
    d = minform.date_input(
        "Please select the Time Period for analysis",
        value = (d[0], d[1]),
        min_value = datetime.date(2001, 1, 1),
        max_value = datetime.date.today(), key = "daterange2")
    find_min = minform.form_submit_button(label='Find Minimums')

with sidebar.expander("Expand for Finding SIP"):
    sipform = st.form(key="invest")
    stock_selector = sipform.selectbox(
        "Select a Ticker",
        all_stocks, index = ss_idx)
    sipdates = sipform.date_input(
        "Please select the Time Period for SIP",
        value = (d[0], d[1]),
        min_value = datetime.date(2001, 1, 1),
        max_value = datetime.date.today())
    sip = sipform.selectbox('SIP Amount', range(10000, 100000, 10000))
    sipsubmit = sipform.form_submit_button(label='Invest Now!')

if data_peek:
    st.markdown(f"Currently Selected Ticker: {stock_selector}")
    st.write('Currently Selected Time Period: ', d[0].strftime("%A - %d %B, %Y"), ' to ', d[1].strftime("%A - %d %B, %Y"))
    data = get_stock_data(stock_selector, d[0], d[1])
    st.markdown(f"Currently Selected trend level: {trend_level}")
    trend_kwds = {"Daily": "1D", "Weekly": "1W", "Monthly": "1M", "Quarterly": "1Q", "Yearly": "1Y"}
    trend_data = data.copy()
    trend_data = trend_data.resample(trend_kwds[trend_level], on='date', origin='start').agg(
        {"open": "first",
         "close": "last",
         "low": "min",
         "high": "max",
         }
        ).dropna()[['open', 'high', 'low', 'close']].reset_index()
    if show_data:
        st.markdown(f"The data is shown below:")
        st.dataframe(trend_data)
    if chart_selector == 'Candle Stick':
        fig = go.Figure(data=[go.Candlestick(x=trend_data['date'],
                open=trend_data['open'],
                high=trend_data['high'],
                low=trend_data['low'],
                close=trend_data['close'])])
        st.plotly_chart(fig, use_container_width=False)
    else:
        chart_dict = {'Line':'line','Bar':'bar'}
        fig=trend_data.iplot(kind=chart_dict[chart_selector], asFigure=True, xTitle="Date", yTitle="Values", x="date", y="close",
                             title=f"{trend_level} chart of {stock_selector}")
        st.plotly_chart(fig, use_container_width=False)

if find_min:
    data = get_stock_data(stock_selector, d[0], d[1])
    stock_data = data.copy()
    date_column = "date"
    value_column = "high"
    rawdf = stock_data.copy()
    rawdf[date_column] = pd.to_datetime(rawdf.date)
    rawdf['year'] = pd.DatetimeIndex(rawdf[date_column]).year
    rawdf['month'] = pd.DatetimeIndex(rawdf[date_column]).month
    rawdf['day'] = pd.DatetimeIndex(rawdf[date_column]).day
    grp = pd.DataFrame(rawdf.groupby(["year","month"])[value_column].count())
    grp = grp.reset_index()
    yearlist = []
    monthlist = []
    valuelist = []
    for index, row in grp.iterrows():
      yearlist.append(str(row["year"]))
      monthlist.append(str(row["month"]))
      valuelist.append(rawdf[(rawdf["year"] == row["year"]) & (rawdf["month"] == row["month"])][value_column].tolist())
    mindf = pd.DataFrame({"year":yearlist, "month":monthlist,"values":valuelist})
    mindf['MinValue'] = [min(x) for x in mindf["values"].tolist()]
    rawdf["year"] = rawdf.year.astype(float)
    mindf["year"] = mindf.year.astype(float)
    rawdf["month"] = rawdf.month.astype(float)
    mindf["month"] = mindf.month.astype(float)
    rawdf[value_column] = rawdf[value_column].astype(float)
    mindf["MinValue"] = mindf.MinValue.astype(float)
    rawdf["day"] = rawdf.day.astype(float)
    mindates = pd.merge(mindf, rawdf,  how='left', left_on=['year','month','MinValue'], right_on = ['year','month',value_column])
    mindates["DateModMon"] = pd.to_datetime(mindates['date']).dt.strftime("%d %b, %Y")
    yearlist = list(mindf.year.unique().astype(int)) 
    grptwo = pd.DataFrame(mindates.groupby(["month","day"], as_index=False).size().reset_index())
    grptwo["date"] = pd.to_datetime("2022-"+grptwo["month"].astype(int).astype(str)+"-"+grptwo["day"].astype(int).astype(str))
    grptwo.set_index('date', inplace = True)
    import calplot
    ax = calplot.calplot(grptwo["size"], cmap = 'Wistia', textformat  ='{:.0f}', figsize = (12, 6), colorbar = False)
    ax[1][0].axes.yaxis.set_visible(False)
    plt.title("Dates with minimum values")
    fig = ax[1][0].get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    st.pyplot(fig)
    yearlist = list(mindf.year.unique().astype(int))
    figheight = 5 * (len(yearlist) // 2)   
    yearlist = list(mindf.year.unique().astype(int))
    pltcnt = (len(yearlist) // 2) + 1
    fig, axs = plt.subplots(pltcnt,2, facecolor='w', edgecolor='k', figsize=(20, figheight))
    axs = axs.ravel()
    i = 0
##    for year in yearlist:
##        ax = sns.barplot(y = "DateModMon", x = "MinValue", data = mindates[mindates["year"]==year], ax = axs[i])
##        mintemp = min(list(mindates[mindates["year"]==year]["MinValue"])) 
##        mintemp = round(mintemp - (mintemp*0.02))
##        maxtemp = max(list(mindates[mindates["year"]==year]["MinValue"])) 
##        maxtemp = round(maxtemp + (maxtemp*0.02))
##        axs[i].set_xlim(mintemp,maxtemp)
##        axs[i].set_xlabel("Price - Month's Minimum Value")
##        axs[i].set_ylabel("Date")
##        axs[i].set_title("Year : "+str(year))
##        ax.bar_label(ax.containers[0])
##        i += 1
##        fig.tight_layout()
##    for ax in axs.flat[len(yearlist):]:
##        ax.remove()
##    st.header("The following plot shows year wise representation of minimum values")
##    st.pyplot(fig)
##    monthlist = [1,2,3,4,5,6,7,8,9,10,11,12]
##    fig, axs = plt.subplots(6,2, figsize=(20, figheight*0.8), facecolor='w', edgecolor='k')
##    axs = axs.ravel()
##    i = 0
##    for month in monthlist:
##        ax = sns.barplot(y = "DateModMon", x = "MinValue", data = mindates[mindates["month"]==month], ax = axs[i])
##        mintemp = min(list(mindates[mindates["month"]==month]["MinValue"])) 
##        mintemp = round(mintemp - (mintemp*0.02))
##        maxtemp = max(list(mindates[mindates["month"]==month]["MinValue"])) 
##        maxtemp = round(maxtemp + (maxtemp*0.02))
##        axs[i].set_xlim(mintemp,maxtemp)
##        axs[i].set_xlabel("Minimum Value")
##        axs[i].set_ylabel("Date")
##        axs[i].set_title("Month "+str(month))
##        ax.bar_label(ax.containers[0])
##        i += 1
##        fig.tight_layout()
##    st.header("Let us now have a look at the month wise minimum values")
##    st.pyplot(fig)

if sipsubmit:
    rawdf = get_data(stock_selector, start_date=sipdates[0], end_date=sipdates[1], index_as_date = False, interval="1d")
    rawdf['date'] = pd.to_datetime(rawdf.date)
    rawdf.set_index('date', inplace = True)
    idx = pd.date_range(rawdf.index.min(), rawdf.index.max())
    completedf = rawdf.reindex(idx, method='bfill')
    completedf['year'] = completedf.index.year
    completedf['month'] = completedf.index.month
    completedf['day'] = completedf.index.day
    cdf = completedf.copy()
    #pd.date_range(cdf.index.min(), cdf.index.max()).difference(cdf.index)
    cdf =cdf[["high","year","month","day"]]
    selected_date_list = []
    total_qty_list =[]
    inv_value_list = []
    tot_inv_list = []
    from_year = sipdates[0].year
    to_year = sipdates[1].year
    cdf = cdf.loc[sipdates[0]:sipdates[1]]
    for d in range(1,29):
      cdftemp = cdf[cdf["day"]==d]
      cdftemp["qty"] = sip//round(cdftemp["high"],2)
      cdftemp["invamt"] = cdftemp["qty"]*cdftemp["high"]
      selected_date_list.append(d)
      total_qty_list.append(int(np.sum(cdftemp["qty"])))
      tot_inv_list.append(round(np.sum(cdftemp["invamt"]),2))
    returnsdf = pd.DataFrame({"Date":selected_date_list, "Total Invested":tot_inv_list, "Quantity":total_qty_list})
    #returnsdf["Total Invested"] = returnsdf["Total Invested"]/100000
    returnsdf["Current Value"] = returnsdf.Quantity*completedf.iloc[-1,completedf.columns.get_loc("low")]
    #returnsdf["Current Value"] = returnsdf["Current Value"]/100000
    returnsdf["Returns(in %)"] = round((((returnsdf["Current Value"] - returnsdf["Total Invested"]) / returnsdf["Total Invested"])*100),2)
    if to_year - from_year > 0:
        returnsdf["CAGR"] = round(returnsdf["Returns(in %)"]*(1/int(to_year - from_year)),2)
    elif to_year - from_year == 0:
        returnsdf["CAGR"] = round(returnsdf["Returns(in %)"],2)
    fig=returnsdf.iplot(kind='line', asFigure=True, xTitle="Day of Month", yTitle="CAGR", x="Date", y="CAGR",
                         title=f"Returns chart of {stock_selector}")
    st.plotly_chart(fig, use_container_width=False)
    st.table(returnsdf)
    
