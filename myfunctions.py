# libs
import streamlit as st
import numpy as np
import plotly.express as px

######## Functions ############

def shift(arr, num, fill_value=np.nan): # shift decay
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result  

@st.cache_data
def draw_img(im):
    fig = px.imshow(im)
    st.plotly_chart(fig)

@st.cache_data
def convert_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

def lifetme_spectro_upload(file_name, file_type):
    if file_type == 'QuTAG MC':
        data = np.loadtxt(file_name, skiprows = 5, delimiter = ';') 
    elif file_type == 'TDC7200':
        data = np.loadtxt(file_name, skiprows = 10, delimiter = '\t') 
    elif file_type == 'PicoHarp':
        data = np.loadtxt(file_name, skiprows = 0, delimiter = '\t') # TODO: PicoHarp compatible
    return data

######## Class fluorescnce decay ############

class fl_decay(): # class for storing lifetime spectro data
    def __init__(self, file_name, decay_data):
        self.file_name = file_name
        self.decay_data = decay_data
        self.raw_data = decay_data
        self.bg = np.average(self.decay_data[0:20]) # first 20 bin = BG window 
        self.norm_decay_data = (self.decay_data-self.bg)/max(self.decay_data-self.bg)
        self.peak = np.argmax(self.decay_data)  
        self.result = np.empty_like(decay_data)
        self.best_fit = np.empty_like(decay_data)
    def update(fl_decay):
        fl_decay.norm_decay_data = (fl_decay.decay_data-fl_decay.bg)/max(fl_decay.decay_data-fl_decay.bg)
    def over_sample(fl_decay): #TODO: not working
        nmax = 4
        fl_decay.decay_data /= np.max(np.array([sum(fl_decay.decay_data[i:i+nmax]) for i in range(0, len(fl_decay.decay_data), nmax)])) // nmax

######## Class FLIM  ############

class FLIM(): # class for storing FLIM data
    def __init__(self, file_name, TCSPC, h, step_x, step_y, length_x, length_y, expo_time):
        self.file_name = file_name
        self.TCSPC = TCSPC
        self.h = h
        self.step_x = step_x
        self.step_y = step_y
        self.length_x = length_x
        self.length_y = length_y
        self.expo_time = expo_time
        self.time_bin = (self.TCSPC).shape[1]
        self.intensity = np.sum(self.TCSPC, axis=1)
        self.intensity_img = np.reshape(self.intensity,(100,100))