# libs
import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

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
        data_temp = np.loadtxt(file_name, skiprows = 5, delimiter = ';') 
        data = data_temp[:,1]
        h = (data_temp[1,0]-data_temp[0,0])*1000000000
    elif file_type == 'TDC7200':
        data_temp = np.loadtxt(file_name, skiprows = 10, delimiter = '\t') 
        data = data_temp[:,1]
        h = 0.055
    elif file_type == 'PicoHarp':
        data = np.loadtxt(file_name, skiprows = 0) # TODO: PicoHarp compatible
        h = 0.004
    return data, h

def check_h(decay,tot_file):
    chk = False
    h = decay[0].h
    for i in range(tot_file):
        if decay[i].h != h:
            chk = True
            break
    return chk

def draw_result(lmfit_result_items,h):
    A = np.array([])
    name_A = np.array([])
    tau = np.array([])
    name_tau = np.array([])
    for name, param in lmfit_result_items:
        if 'A' in name:
            A = np.append(A,param.value)
            name_A = np.append(name_A,name)
        elif 'tau' in name:
            tau = np.append(tau,param.value*h)
            name_tau = np.append(name_tau,name)
    col1, col2 = st.columns(2)
    with col1:
        # pie amplitude
        df = pd.DataFrame(data=A,columns=['A'])
        fig = px.pie(df, values='A',names=name_A,width=300,color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig)
    with col2:    
        # bar tau
        df2 = pd.DataFrame(data=tau,columns=['tau'],index=name_tau)
        fig = px.bar(df2, y='tau',width=300,color_discrete_sequence=px.colors.qualitative.Pastel_r)
        st.plotly_chart(fig)    

def irf_select(decay):
    name = np.array([])
    for i in range(len(decay)):
        name = np.append(name,decay[i].file_name)
    data_df = pd.DataFrame(
        {
            "favorite": [True, False],
            "widgets": name,
        }
    )
    irf_select = st.data_editor(
        data_df,
        column_config={
            "favorite": st.column_config.CheckboxColumn(
                default=False,
            )
        },
        disabled=["widgets"],
        hide_index=True,
    )
    if sum(irf_select['favorite']) > 1:
        st.error('Only one IRF can be slected')
        irf_select['favorite'] = False
        st.stop()
    else:
        irf = np.array((irf_select['widgets'][irf_select['favorite']==True]))
        st.write(irf)
    return irf_index

######## Class fluorescnce decay ############

class fl_decay(): # class for storing lifetime spectro data
    def __init__(self, file_name, decay_data,h):
        self.file_name = file_name
        self.decay_data = decay_data
        self.raw_data = decay_data
        self.bg = np.average(self.decay_data[0:20]) # first 20 bin = BG window 
        self.norm_decay_data = (self.decay_data-self.bg)/max(self.decay_data-self.bg)
        self.peak = np.argmax(self.decay_data)  
        self.result = np.empty_like(decay_data)
        self.best_fit = np.empty_like(decay_data)
        self.h = h
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