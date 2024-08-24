# libs
import streamlit as st
import numpy as np
import math
import pandas as pd
import scipy.io as sio
import plotly.express as px

# User file
import myfunctions
import FLIM_analysis

st.title("FLIM")
st.markdown('''***Required variables***  
TCSPC - 2d array (row = pixel, col = time bin)  
h - bin width (ns)  
step_x step_y - num of pixels along x and y axes  
length_x length_y - real distance (μm)
expo_time - exposure time (ms)''')
#TODO: update the save button GUI to match this file standard

#### Upload area####
file_type = st.selectbox("Select File Type",['.mat','.h5'])
uploaded_file = st.file_uploader("Choose FLIM data", accept_multiple_files=False) # create upload box
if uploaded_file:
    FLIM_data = []
    if file_type == '.mat':
        data = sio.loadmat(uploaded_file)
        # st.write(data.keys()) # print all variables to check
        try:
            FLIM_data.append(myfunctions.FLIM(file_name=uploaded_file.name,TCSPC=np.array(data['TCSPC']),h=np.array(data['h']),step_x=np.array(data['step_x']),step_y=np.array(data['step_y']),length_x=np.array(data['length_x']),length_y=np.array(data['length_y']),expo_time=np.array(data['expo_time'])))
        except:
            st.error('Error in loading file')
            st.stop()
        # lifetimes = FLIM_analysis.calculate_lifetimes_lmfit(FLIM_data[0].TCSPC, np.arange(0,FLIM_data[0].time_bin,1))
        myfunctions.draw_img(FLIM_data[0].intensity_img)

        #### Crop Data ####
        start_bin, stop_bin = st.slider('Select fitting window', 0,FLIM_data[0].time_bin,(0,FLIM_data[0].time_bin))
        # for decay_id in range(tot_file):
        #     decay[decay_id].decay_data = decay[decay_id].decay_data[start_bin:stop_bin]
        #     data_len = stop_bin-start_bin
        #     decay[decay_id].update()   
        st.line_chart((np.sum(FLIM_data[0].TCSPC,axis=0)[start_bin:stop_bin]))

     #TODO continue work from here