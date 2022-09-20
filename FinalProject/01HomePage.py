import streamlit as st

st.set_page_config(page_title="ML for Time Series Forecasting", page_icon="📈", layout="wide", initial_sidebar_state="collapsed", menu_items=None)

with st.sidebar:
    st.write("Above page links can be used to navigate around the application")

st.title("Application of Machine Learning Models for Time Series Analysis")
st.caption("This interactive dashboard has been developed to apply Statistical and Machine Learning models on historical stock prices for predicting future prices/trends")

tab1, tab2, tab3, tab4 = st.tabs(["About", "See Model Results", "Make Predictions", "Contact"])

with tab1:
    st.markdown(
        """
        This application aims at training machine learning models using historical stock prices of a selected stock in an attempt to predict future price movements and trends.
        """)
    with st.expander("Statistical Models", expanded = False):
        st.markdown(
            """
            The models implemented in this research are:
            - **ARIMA**:
                - **A**uto **R**egressive **I**ntegrated **M**oving **A**verage
                - The most basic model that has been implemented for a long time for time series forecasting.
                - Different variants are available to account for Seasonality and Trends in data.
                - Works well with Univariate data.
            - **LSTM**:
                - **L**ong **S**hort **T**erm **M**emory
                - Advanced Deep Learning model that is primarily being used for text translation by Google, FB and many other organizations.
                - Works well with complex input and multivariate analysis.
            - **FBProphet**:
                - Facebook's in-house time series forecasting model.
                - Developed and tuned with accounting for recent developments in statistical modelling techniques.
                - Was primarily developed for "producing reliable forecasts for planning and goal setting".
            """
        )
    with st.expander("Application Navigation", expanded= False):
        st.markdown(
            """
            This application primarily consists of two pages:
            - **Model Results** - This page can be used to view the model training parameters, model descriptions, statistical analysis and their respective results.
            - **Make Predictions** - This page can be used to make and view future predictions based on trained models.
            The pages can be viewed and navigated by expanding the menu that can be found towards the left of this application. More about the pages can be found in the adjacent tabs of this page.
            """
        )

with tab2:
    st.markdown(
        """
        - "Model Results" page can be used to view the trained models and their results.
        - Predefined Metrics can be used to view the results instantly for the already trained models. The training metrics are listed in the page for reference.
        - The different specifications displayed are:
            - Model definition/description
            - Train / Test data results
            - Performance Metrics

        """
    )

with tab3:
    st.markdown(
        """
        - "Make Predictions" page can be used to predict future prices using all three models.
        - Inputs are:
            - Ticker/Stock Name for which you would like to see predictions.
            - Number of days to predict in future
        - The graphs with appropriate predictions and results are then displayed.
        """
    )
    st.warning("It is important to note that the model is trained at runtime and hence, will take longer than usual for predictions")

with tab4:
    st.markdown(
                """
                For any queries/feedback, please feel free to get in touch with me using the following details:
                - **LinkedIn** - <a href="https://www.linkedin.com/in/shreyashrangarajan"> Shreyash Rangarajan </a>
                - **Personal Digital Portfolio** - <a href="https://shreyashrangarajan.wixsite.com/shreyash"> Shreyash's Memoir </a>
                - **E-Mail ID** - shreyashrangarajan@gmail.com
                """, unsafe_allow_html=True)

    