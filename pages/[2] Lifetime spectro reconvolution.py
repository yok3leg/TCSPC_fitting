# libs
import streamlit as st
import numpy as np
import math
import pandas as pd
import scipy.io as sio
import plotly.express as px

# User file
import lifetime_spectro_analysis
import myfunctions

st.title("FLIM")
st.markdown("This analysis uses Liftfit (https://lifefit.readthedocs.io/en/latest/index.html)")

#### Upload area####
file_type = st.selectbox("Select File Type",['QuTAG MC','TDC7200','PicoHarp'])
uploaded_files = st.file_uploader("Choose a txt file", accept_multiple_files=True) # create upload box
if uploaded_files: # if there is/are file(s) uploaded
    n_components = st.sidebar.number_input("Number of decay components", value=1,step=1,max_value=4,min_value=1) # select number of exp component
    tot_file = len(uploaded_files)
    # st.write(str(tot_file)+' files uploaded') # for checking total files
    decay = [] #
    try:
        for file_name in uploaded_files: # read all files
            data = myfunctions.lifetme_spectro_upload(file_name, file_type)
            decay.append(myfunctions.fl_decay(file_name=file_name.name, decay_data=data[:,1])) # store in the fl_decay class
        data_len = len(data[:,1])
    except:
        st.error('File type error')
        st.stop()

    