import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import calmap
import calplot
from plotly_calplot import calplot

st.write('matplotlib: {}'.format(matplotlib.__version__))
srs = "I Am Loving It!"
st.write("SRS Welcomes you!",srs)
