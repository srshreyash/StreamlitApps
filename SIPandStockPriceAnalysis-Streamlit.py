import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.graph_objects as go

# set_page_config must be the first Streamlit call in the script
st.set_page_config(page_title="Stock Price and SIP analysis", page_icon="🤘", layout="wide", initial_sidebar_state="collapsed", menu_items=None)

# NIFTY 50 constituents (Yahoo symbols). yahoo_fin's tickers_nifty50() scraped a page that
# no longer exists, so the list is kept here. Index membership changes a few times a year,
# so edit this list when NSE rebalances.
NIFTY50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "MAXHEALTH.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "TMPV.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Fetching prices from Yahoo Finance...")
def get_stock_data(ticker, start_date, end_date):
    # yfinance's end date is exclusive, so add a day to include the selected end date
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date + datetime.timedelta(days=1),
                                     interval="1d", auto_adjust=False)
    if hist.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    hist = hist.reset_index()
    hist.columns = [str(c).strip().lower() for c in hist.columns]
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None).dt.normalize()
    return hist[["date", "open", "high", "low", "close", "volume"]]


def load_ticker_data(ticker, start_date, end_date):
    # Shows a readable error instead of silently blanking the page
    try:
        data = get_stock_data(ticker, start_date, end_date)
    except Exception as exc:
        st.error(f"Could not download data for {ticker}: {exc}")
        st.stop()
    if data.empty:
        st.error(f"Yahoo Finance returned no data for {ticker} between {start_date} and {end_date}.")
        st.stop()
    return data


def filter_user_data(df, start_date, end_date):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]


def line_or_bar(df, kind, x, y, title, x_title, y_title):
    trace = go.Scatter(x=df[x], y=df[y], mode="lines", name=y) if kind == "Line" else go.Bar(x=df[x], y=df[y], name=y)
    fig = go.Figure(data=[trace])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return fig


def require_range(dates):
    # date_input returns a 1-tuple while the user has only picked the start date
    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.warning("Please select both a start and an end date.")
        st.stop()
    return dates


st.title("Stock Price Analysis - for SIP purposes")
st.caption("This app has been designed to answer the question 'Is there any particular date/week for SIP which yields higher returns than others?' Please use the left sidebar to navigate and explore the application.")
tab1, tab2, tab3 = st.tabs(["Read Me", "Assumptions and Constraints", "Contact"])
with tab1:
    with st.expander("All About this application", expanded=True):
        st.markdown(
            """
            Hi, Thank you for checking out my aplication. Please find the features of this app below. There is a sidebar towards the left of the application that can be expanded to view the following options -
             - **Choice of Input**:
                - Your own input:
                    - This option can be checked to upload your own csv file with all the data in it. It is advisable to download the files from the <a href="https://finance.yahoo.com/"> Official website of yahoo finance </a>.
                    - The application has been built on yahoo's historical data template and will automatically pick up the start and end dates from the given csv file.
                    - For data from other sources, Please make sure that the following column names are followed:
                        [date, high, close, low, open, volume].
                        Please note that the column names are not case sensitive and can be in any order.
                - Select ticker from dropdown:
                    - Another option is to use the dropdown in the sidebar to select your favourite ticker.
                    - The dropdown lists the NIFTY 50 stocks, and their prices are fetched at runtime from Yahoo Finance.
            - **Peek into the data**:
                - View the prices of tickers in different charts such as Line, Bar and Candle Sticks.
                - Choose your own time period for analysis and visualization.
                - An option is also given to view the raw data in the form of a table along with the chart.
            - **Find dates of occurence of monthly minimums**:
                - This is my favourite part and core of the application.
                - View the dates of occurence of minimum value of the selected ticker every month throughout the selected period.
                - This will answer whether there is any pattern to the price movement and if the stock is cheapest at or around any particular date every month.
            - **Find SIP Value**:
                - Simulate your SIP returns over the last twenty years.
                - Calculates and compares SIP executed on same day of every month throughout the period and generates graphs and the table with relevant data.

            """, unsafe_allow_html=True)
        st.warning("**_IMPORTANT_ : Please note that this is just an exhibition and application of my skills/knowledge and is in no manner to be construed as an investment advise. Any investment decision made based on this analysis should be at one's own risk. This is developed strictly for educational purposes only.**")
with tab2:
    with st.expander("Assumptions and Constraints", expanded=True):
        st.markdown(
            """
            1. Data has been fetched from Yahoo Finance via the yfinance python library. This data is maintained and updated regularly by yahoo finance.
            2. Missing data (including weekends) has been filled using the available values of the immediately following trading day for which data is available, in order to keep the analysis uniform.
            3. For finding minimums, the high value of every day has been considered in order to find minimums of every month.
            4. For SIP, it has been considered that investment is made on the high price of the instrument on that particular day. This is for practical purposes.

            """)
with tab3:
    with st.expander("Contact Details", expanded=True):
        st.markdown(
            """
            For any queries/feedback, please feel free to get in touch with me using the following details:
            - **LinkedIn** - <a href="https://www.linkedin.com/in/shreyashrangarajan"> Shreyash Rangarajan </a>
            - **Personal Digital Portfolio** - <a href="https://shreyashrangarajan.wixsite.com/shreyash"> Shreyash's Memoir </a>
            - **E-Mail ID** - shreyashrangarajan@gmail.com
            """, unsafe_allow_html=True)

all_stocks = NIFTY50_TICKERS

sidebarv = st.sidebar
sidebarv.write("Please select the following checkbox if you would like to analyse your own data. Otherwise, please leave it unchecked.")
userfile = sidebarv.checkbox("I have my own data file")

if userfile:
    uploaded_file = sidebarv.file_uploader(label="Please upload the file: ", type=["csv"])
    if uploaded_file is None:
        sidebarv.info("Upload a CSV file to continue.")
        st.stop()
    indf = pd.read_csv(uploaded_file)
    filename = uploaded_file.name[:-4]
    indf.columns = [i.strip().lower() for i in indf.columns.values.tolist()]
    indf["date"] = pd.to_datetime(indf["date"])
    mindate = indf["date"].min().date()
    maxdate = indf["date"].max().date()

if not userfile:
    sidebarv.header("The NIFTY 50 stocks trading in the National Stock Exchange of India are listed below. Please select one for analysis.")

with sidebarv.expander("Peek into the data"):
    form = st.form(key='my_form')
    if userfile:
        stock_selector = form.selectbox("Select a Ticker", [filename], disabled=True)
        d = form.date_input(
            "Please select the Time Period for analysis",
            value=(mindate, maxdate),
            min_value=mindate,
            max_value=maxdate, key="daterangeuser")

    if not userfile:
        stock_selector = form.selectbox("Select a Ticker", all_stocks)
        ss_idx = all_stocks.index(stock_selector)
        d = form.date_input(
            "Please select the Time Period for analysis",
            value=(datetime.date(2001, 1, 1), datetime.date.today()),
            min_value=datetime.date(2001, 1, 1),
            max_value=datetime.date.today(), key="daterange")

    trend_level = form.selectbox("Trend Level", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                                 help="This will group and display data based on your preference")
    chart_selector = form.selectbox("Chart Type: ", ('Line', 'Bar', 'Candle Stick'),
                                    help="Select your favourite chart type for visualizing data. My Personal Fav: Candlestick ❤️‍🔥")
    show_data = form.checkbox("Show Data", key="showdata")
    data_peek = form.form_submit_button(label='Peek into the data')

# Default range for the other two forms follows the first one, even mid-selection
default_range = tuple(d) if isinstance(d, (tuple, list)) and len(d) == 2 else (datetime.date(2001, 1, 1), datetime.date.today())

with sidebarv.expander("Find dates of occurence of monthly Minimums"):
    minform = st.form(key='min_form')
    if userfile:
        min_stock = minform.selectbox("Select a Ticker", [filename], disabled=True, key="minticker")
        min_d = minform.date_input(
            "Please select the Time Period for analysis",
            value=(mindate, maxdate),
            min_value=mindate,
            max_value=maxdate, key="daterangeuser2")
    if not userfile:
        min_stock = minform.selectbox("Select a Ticker", all_stocks, index=ss_idx, key="minticker")
        min_d = minform.date_input(
            "Please select the Time Period for analysis",
            value=default_range,
            min_value=datetime.date(2001, 1, 1),
            max_value=datetime.date.today(), key="daterange2")
    find_min = minform.form_submit_button(label='Find Minimums')

with sidebarv.expander("Find SIP value"):
    sipform = st.form(key="invest")
    if userfile:
        sip_stock = sipform.selectbox("Select a Ticker", [filename], disabled=True, key="sipticker")
        sipdates = sipform.date_input(
            "Please select the Time Period for SIP",
            value=(mindate, maxdate),
            min_value=mindate,
            max_value=maxdate, key="sipdatesuser")
    if not userfile:
        sip_stock = sipform.selectbox("Select a Ticker", all_stocks, index=ss_idx, key="sipticker")
        sipdates = sipform.date_input(
            "Please select the Time Period for SIP",
            value=default_range,
            min_value=datetime.date(2001, 1, 1),
            max_value=datetime.date.today(), key="sipdates")

    sip = sipform.selectbox('SIP Amount', range(10000, 100000, 10000))
    sipsubmit = sipform.form_submit_button(label='Invest Now!')

if data_peek:
    d = require_range(d)
    st.text("-------------------------------------------------------------------------------------------")
    st.header("Data Visualization:")
    st.markdown(f"_Selected Ticker_ : **{stock_selector}**")
    d_zero = d[0].strftime("%A - %d %B, %Y")
    d_one = d[1].strftime("%A - %d %B, %Y")
    st.markdown(f'_Selected Time Period_ : **{d_zero} to {d_one}**')
    if userfile:
        trend_data = filter_user_data(indf, d[0], d[1])
    if not userfile:
        trend_data = load_ticker_data(stock_selector, d[0], d[1])
    st.markdown(f"_Selected trend level_ : **{trend_level}**")
    # pandas 2.2+ renamed month/quarter/year-end aliases to ME/QE/YE; the old ones now raise
    trend_kwds = {"Daily": "1D", "Weekly": "1W", "Monthly": "1ME", "Quarterly": "1QE", "Yearly": "1YE"}
    trend_data = trend_data.resample(trend_kwds[trend_level], on='date', origin='start').agg(
        {"open": "first",
         "close": "last",
         "low": "min",
         "high": "max",
         }
        ).dropna()[['open', 'high', 'low', 'close']].reset_index()
    if chart_selector == 'Candle Stick':
        fig = go.Figure(data=[go.Candlestick(x=trend_data['date'],
                open=trend_data['open'],
                high=trend_data['high'],
                low=trend_data['low'],
                close=trend_data['close'], )])
        fig.update_layout(title=f"{trend_level} {chart_selector} chart of {stock_selector}", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, width="content")
    else:
        fig = line_or_bar(trend_data, chart_selector, "date", "close",
                          f"{trend_level} chart of {stock_selector}", "Date", "Values")
        st.plotly_chart(fig, width="content")
    if show_data:
        st.markdown("The raw data is shown below:")
        st.dataframe(trend_data)

if find_min:
    min_d = require_range(min_d)
    st.text("-------------------------------------------------------------------------------------------")
    st.header("Data Analysis - Finding Minimums:")
    if userfile:
        data = filter_user_data(indf, min_d[0], min_d[1])
    if not userfile:
        data = load_ticker_data(min_stock, min_d[0], min_d[1])
    st.markdown(f"**The following visual shows the occurence of minimum value of {min_stock} seen every month between {min_d[0]} and {min_d[1]}**")
    stock_data = data.copy()
    date_column = "date"
    value_column = "high"
    rawdf = stock_data.copy()
    rawdf[date_column] = pd.to_datetime(rawdf.date)
    rawdf['year'] = pd.DatetimeIndex(rawdf[date_column]).year
    rawdf['month'] = pd.DatetimeIndex(rawdf[date_column]).month
    rawdf['day'] = pd.DatetimeIndex(rawdf[date_column]).day
    grp = pd.DataFrame(rawdf.groupby(["year", "month"])[value_column].count())
    grp = grp.reset_index()
    yearlist = []
    monthlist = []
    valuelist = []
    for index, row in grp.iterrows():
        yearlist.append(str(row["year"]))
        monthlist.append(str(row["month"]))
        valuelist.append(rawdf[(rawdf["year"] == row["year"]) & (rawdf["month"] == row["month"])][value_column].tolist())
    mindf = pd.DataFrame({"year": yearlist, "month": monthlist, "values": valuelist})
    mindf['MinValue'] = [min(x) for x in mindf["values"].tolist()]
    rawdf["year"] = rawdf.year.astype(float)
    mindf["year"] = mindf.year.astype(float)
    rawdf["month"] = rawdf.month.astype(float)
    mindf["month"] = mindf.month.astype(float)
    rawdf[value_column] = rawdf[value_column].astype(float)
    mindf["MinValue"] = mindf.MinValue.astype(float)
    rawdf["day"] = rawdf.day.astype(float)
    mindates = pd.merge(mindf, rawdf, how='left', left_on=['year', 'month', 'MinValue'], right_on=['year', 'month', value_column])
    mindates["DateModMon"] = pd.to_datetime(mindates['date']).dt.strftime("%d %b, %Y")
    grptwo = pd.DataFrame(mindates.groupby(["month", "day"], as_index=False).size().reset_index())
    grptwo["date"] = pd.to_datetime("2022-" + grptwo["month"].astype(int).astype(str) + "-" + grptwo["day"].astype(int).astype(str))
    grptwo.set_index('date', inplace=True)
    try:
        import calplot
        ax = calplot.calplot(grptwo["size"], cmap='Wistia', textformat='{:.0f}', figsize=(12, 6), colorbar=False)
        ax[1][0].axes.yaxis.set_visible(False)
        plt.title("Dates with minimum values")
        fig = ax[1][0].get_figure()
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        st.pyplot(fig)
    except ImportError:
        st.warning("Install the calplot package to see the calendar view of minimum dates.")
    st.write("Note: Everyday's high value has been considered for finding the minimum value in each month")
    yearlist = list(mindf.year.unique().astype(int))
    pltcnt = (len(yearlist) // 2) + 1
    figheight = 5 * pltcnt
    fig, axs = plt.subplots(pltcnt, 2, facecolor='w', edgecolor='k', figsize=(20, figheight))
    axs = axs.ravel()
    i = 0
    st.markdown("**The following plot shows year wise representation of minimum values**")
    progress_bar = st.progress(0)
    prog_cnt = 0
    for year in yearlist:
        ax = sns.barplot(y="DateModMon", x="MinValue", data=mindates[mindates["year"] == year], ax=axs[i])
        mintemp = min(list(mindates[mindates["year"] == year]["MinValue"]))
        mintemp = round(mintemp - (mintemp * 0.02))
        maxtemp = max(list(mindates[mindates["year"] == year]["MinValue"]))
        maxtemp = round(maxtemp + (maxtemp * 0.02))
        axs[i].set_xlim(mintemp, maxtemp)
        axs[i].set_xlabel("Price - Month's Minimum Value")
        axs[i].set_ylabel("Date")
        axs[i].set_title("Year : " + str(year))
        ax.bar_label(ax.containers[0])
        i += 1
        prog_cnt = min(100, prog_cnt + int(100 / len(yearlist)))
        progress_bar.progress(prog_cnt)
    fig.tight_layout()
    progress_bar.empty()
    for ax in axs.flat[len(yearlist):]:
        ax.remove()
    st.info("If you can see the running icon on top right corner of the webpage, please sit tight. It means that the charts are loading.")
    st.pyplot(fig)
    monthlist = [m for m in range(1, 13) if (mindates["month"] == m).any()]
    fig, axs = plt.subplots(6, 2, figsize=(20, max(figheight * 0.8, 24)), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    i = 0
    st.markdown("**Let us now have a look at the month wise minimum values**")
    progress_bar = st.progress(0)
    prog_cnt = 0
    for month in monthlist:
        ax = sns.barplot(y="DateModMon", x="MinValue", data=mindates[mindates["month"] == month], ax=axs[i])
        mintemp = min(list(mindates[mindates["month"] == month]["MinValue"]))
        mintemp = round(mintemp - (mintemp * 0.02))
        maxtemp = max(list(mindates[mindates["month"] == month]["MinValue"]))
        maxtemp = round(maxtemp + (maxtemp * 0.02))
        axs[i].set_xlim(mintemp, maxtemp)
        axs[i].set_xlabel("Minimum Value")
        axs[i].set_ylabel("Date")
        axs[i].set_title("Month " + str(month))
        ax.bar_label(ax.containers[0])
        i += 1
        prog_cnt = min(100, prog_cnt + int(100 / len(monthlist)))
        progress_bar.progress(prog_cnt)
    fig.tight_layout()
    progress_bar.empty()
    for ax in axs.flat[len(monthlist):]:
        ax.remove()
    st.pyplot(fig)

if sipsubmit:
    sipdates = require_range(sipdates)
    st.text("-------------------------------------------------------------------------------------------")
    st.header("SIP Returns Analysis:")
    st.markdown(f"The following graph shows the returns of SIPs executed on the same dates every month for {sip_stock}: ")
    if userfile:
        rawdf = indf.copy()
    if not userfile:
        rawdf = load_ticker_data(sip_stock, sipdates[0], sipdates[1]).copy()
    rawdf['date'] = pd.to_datetime(rawdf.date)
    rawdf.set_index('date', inplace=True)
    rawdf.sort_index(inplace=True)
    idx = pd.date_range(rawdf.index.min(), rawdf.index.max())
    completedf = rawdf.reindex(idx, method='bfill')
    completedf['year'] = completedf.index.year
    completedf['month'] = completedf.index.month
    completedf['day'] = completedf.index.day
    cdf = completedf[["high", "year", "month", "day"]].copy()
    selected_date_list = []
    total_qty_list = []
    tot_inv_list = []
    from_year = sipdates[0].year
    to_year = sipdates[1].year
    cdf = cdf.loc[pd.Timestamp(sipdates[0]):pd.Timestamp(sipdates[1])]
    for day in range(1, 29):
        cdftemp = cdf[cdf["day"] == day].copy()
        cdftemp["qty"] = sip // round(cdftemp["high"], 2)
        cdftemp["invamt"] = cdftemp["qty"] * cdftemp["high"]
        selected_date_list.append(day)
        total_qty_list.append(int(np.sum(cdftemp["qty"])))
        tot_inv_list.append(round(np.sum(cdftemp["invamt"]), 2))
    returnsdf = pd.DataFrame({"Date": selected_date_list, "Total Invested": tot_inv_list, "Quantity": total_qty_list})
    returnsdf["Current Value"] = returnsdf.Quantity * completedf["low"].iloc[-1]
    returnsdf["Returns(in %)"] = round((((returnsdf["Current Value"] - returnsdf["Total Invested"]) / returnsdf["Total Invested"]) * 100), 2)
    if to_year - from_year > 0:
        returnsdf["CAGR"] = round(returnsdf["Returns(in %)"] * (1 / int(to_year - from_year)), 2)
    else:
        returnsdf["CAGR"] = round(returnsdf["Returns(in %)"], 2)
    fig = line_or_bar(returnsdf, "Line", "Date", "CAGR", f"Returns chart of {sip_stock}", "Day of Month", "CAGR")
    st.plotly_chart(fig, width="content")
    st.markdown("The following table shows the raw data on which the above chart has been plotted. Feel free to sort the columns and analyse as per your interest.")
    st.dataframe(returnsdf)
    st.info("All the values are in ₹ (Indian Rupees - INR)")
