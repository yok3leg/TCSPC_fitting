# libs
import streamlit as st
import numpy as np
import math
import pandas as pd

# user defined function
import myfunctions
import lifetime_spectro_analysis

st.title("Exponential decay lmfit")

#### Upload area####
file_type = st.selectbox("Select File Type",['QuTAG MC','TDC7200','PicoHarp'])
uploaded_files = st.file_uploader("Choose a txt file", accept_multiple_files=True) # create upload box
if uploaded_files: # if there is/are file(s) uploaded
    n_components = st.sidebar.number_input("Number of decay components", value=1,step=1,max_value=4,min_value=1) # select number of exp component
    tot_file = len(uploaded_files)
    # st.write(str(tot_file)+' files uploaded') # for checking total files
    decay = [] #
    # try:
    for file_name in uploaded_files: # read all files
        data, h = myfunctions.lifetme_spectro_upload(file_name, file_type)
        decay.append(myfunctions.fl_decay(file_name=file_name.name, decay_data=data, h=h)) # store in the fl_decay class
    data_len = len(data)
    if tot_file > 1 and myfunctions.check_h(decay=decay,tot_file=tot_file): # check if bin width are identical in all files
        st.error('Bin width are not identical')
        st.stop()
    # except:
    #     st.error('File type error')
    #     st.stop()

    #### Over Sample ####
    en_oversample = st.sidebar.checkbox('Oversample x4 (Not working)') #TODO:
    if en_oversample:
        for decay_id in range(tot_file):
            decay[decay_id].over_sample()
            decay[decay_id].update()

    #### Time alignment ####
    en_shift = st.sidebar.selectbox("Alignment",["None","Align Peak","Edge20","Edge50","Edge80"]) #TODO add mode "Edge20","Edge50","Edge80"
    if en_shift == 'None':
        peak_temp = 0
    elif en_shift == 'Align Peak': 
        peak_temp = math.inf # define peak position
        for decay_id in range(tot_file): # find the most left peak
            if decay[decay_id].peak < peak_temp:
                peak_temp = decay[decay_id].peak 
        for decay_id in range(tot_file): # align every data to most left peak
            decay[decay_id].decay_data = myfunctions.shift(decay[decay_id].decay_data,peak_temp-decay[decay_id].peak, fill_value=0)
            decay[decay_id].update()
    elif en_shift == 'Edge20': 
        peak_temp = 0
    elif en_shift == 'Edge50':
        peak_temp = 0
    elif en_shift == 'Edge80':
        peak_temp = 0

    #### BG window ####
    en_BG_window = st.sidebar.checkbox('Manual BG window', value=False)
    if en_BG_window:
        start_BG_bin, stop_BG_bin = st.slider('Select BG window', min_value=0,max_value=data_len,step=1,value=(0,20))
        plot_bg = np.empty((0,stop_BG_bin-start_BG_bin))
        for decay_id in range(tot_file):
            decay[decay_id].bg = np.average(decay[decay_id].decay_data[start_BG_bin:stop_BG_bin])
            plot_bg = np.vstack((plot_bg,decay[decay_id].decay_data[start_BG_bin:stop_BG_bin]))
            decay[decay_id].update()
        st.line_chart(np.transpose(plot_bg))

    #### Crop Data ####
    start_bin, stop_bin = st.slider('Select fitting window', 0,data_len,(peak_temp,data_len))
    for decay_id in range(tot_file):
        decay[decay_id].decay_data = decay[decay_id].decay_data[start_bin:stop_bin]
        data_len = stop_bin-start_bin
        decay[decay_id].update()       

    ### Fitting ####
    for decay_id in range(tot_file):
        [decay[decay_id].result,decay[decay_id].best_fit] = lifetime_spectro_analysis.fitting(n_components=n_components, y_data=decay[decay_id].norm_decay_data,t=np.arange(0,data_len,1))

    #### Plot result ####
    plot_data = np.empty((0,data_len))
    columns = []
    for decay_id in range(tot_file):
        plot_data = np.vstack((plot_data,decay[decay_id].norm_decay_data))
        columns.append(decay[decay_id].file_name[:-4])
        plot_data = np.vstack((plot_data,decay[decay_id].best_fit))
        columns.append(decay[decay_id].file_name[:-4]+' Fit')
    plot_data = pd.DataFrame(data=np.transpose(plot_data),columns=columns) # convert to dataframe
    csv = myfunctions.convert_csv(plot_data) # convert to UTF8
    st.line_chart(plot_data) # plot data frame
    st.download_button(label='Download fitted data', data=csv, file_name='Fitted_data.csv',mime="text/csv")      

    #### Report result ####
    tab = st.tabs([decay[decay_id].file_name for decay_id in range(tot_file)])
    for decay_id in range(tot_file):
        with tab[decay_id]:
            myfunctions.draw_result(decay[decay_id].result.params.items(),h)
            with st.expander("Fit Report"):
                st.text(decay[decay_id].result.fit_report())
            st.download_button(label='Download '+'Report '+decay[decay_id].file_name, data=decay[decay_id].result.fit_report(),file_name='Report_'+decay[decay_id].file_name)

