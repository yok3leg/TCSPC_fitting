import streamlit as st
import numpy as np
import math
import pandas as pd
import Mode1_exponential_decay_lmfit as Mode1

# Comments
# alignment
# - smooth and find peak
# - add align edge option (20%, 50%, 80%)

# tab
# - data selection for fits - Done

# bg
# - add slider / refresh data - Done

# UX
# - label data

# export
# - csv: t_offset, norm_count, fit_y - Done
# - fit_params from lmfit - Done

# other fitting methods?

############################################## Custom Function ################################################
class fl_decay(): # class for storing data
    def __init__(self, file_name, decay_data):
        self.file_name = file_name
        self.decay_data = decay_data
        self.raw_data = decay_data
        self.bg = np.average(decay_data[0:20]) # first 20 bin = BG window #TODO add slider/reload
        self.norm_decay_data = (decay_data-self.bg)/max(decay_data-self.bg)
        self.peak = np.argmax(self.decay_data)  
        self.result = np.empty_like(decay_data)
        self.best_fit = np.empty_like(decay_data)
    def update(fl_decay):
        fl_decay.norm_decay_data = (fl_decay.decay_data-fl_decay.bg)/max(fl_decay.decay_data-fl_decay.bg)
    def over_sample(fl_decay): #TODO: not working
        nmax = 4
        fl_decay.decay_data /= np.max(np.array([sum(fl_decay.decay_data[i:i+nmax]) for i in range(0, len(fl_decay.decay_data), nmax)])) // nmax

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
def convert_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

########################################### Home Page ############################################

st.title('QLAB Lifetime Tools') # page title
analyze_mode = st.sidebar.selectbox("Select method",["None","Exponential decay lmfit","TDC7200","Coming soon"]) # mode selection

if analyze_mode == "None": # Homepage
    st.markdown("Please select analyzing tool")


########################################### Exponential decay lmfit QuTAG MC ############################################

elif analyze_mode == "Exponential decay lmfit": # Mode1
    st.markdown("***Note:*** Fitted Data must be multiplied with *bin width (ns)*")
    n_components = st.sidebar.number_input("Number of decay components", value=1,step=1,max_value=4,min_value=1) # select number of exp component

    #### Upload area####
    file_type = st.selectbox("Select File Type",['QuTAG MC','TDC7200'])
    uploaded_files = st.file_uploader("Choose a txt file", accept_multiple_files=True) # create upload box
    if uploaded_files: # if there is/are file(s) uploaded
        tot_file = len(uploaded_files)
        # st.write(str(tot_file)+' files uploaded') # for checking total files
        decay = [] #
        for file_name in uploaded_files: # read all files
            if file_type == 'QuTAG MC':
                data = np.loadtxt(file_name, skiprows = 5, delimiter = ';') 
            elif file_type == 'TDC7200':
                data = np.loadtxt(file_name, skiprows = 5, delimiter = ';') # TODO: TDC7200 compatible
            decay.append(fl_decay(file_name=file_name.name, decay_data=data[:,1])) # store in the fl_decay class
        data_len = len(data[:,1])

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
                decay[decay_id].decay_data = shift(decay[decay_id].decay_data,peak_temp-decay[decay_id].peak, fill_value=0)
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
            [decay[decay_id].result,decay[decay_id].best_fit] = Mode1.fitting(n_components=n_components, y_data=decay[decay_id].norm_decay_data,t=np.arange(0,data_len,1))

        #### Plot result ####
        plot_data = np.empty((0,data_len))
        columns = []
        for decay_id in range(tot_file):
            plot_data = np.vstack((plot_data,decay[decay_id].norm_decay_data))
            columns.append(decay[decay_id].file_name[:-4])
            plot_data = np.vstack((plot_data,decay[decay_id].best_fit))
            columns.append(decay[decay_id].file_name[:-4]+' Fit')
        plot_data = pd.DataFrame(data=np.transpose(plot_data),columns=columns) # convert to dataframe
        csv = convert_csv(plot_data) # convert to UTF8
        st.line_chart(plot_data) # plot data frame
        st.download_button(label='Download fitted data', data=csv, file_name='Fitted_data.csv',mime="text/csv")      

        #### Report result ####
        tab = st.tabs([decay[decay_id].file_name for decay_id in range(tot_file)])
        for decay_id in range(tot_file):
            with tab[decay_id]:
                st.text(decay[decay_id].result.fit_report())
                st.download_button(label='Download '+'Report '+decay[decay_id].file_name, data=decay[decay_id].result.fit_report(),file_name='Report_'+decay[decay_id].file_name)

################################################### TDC7200 ####################################################

elif analyze_mode == "TDC7200":
    st.markdown("Coming soon")

elif analyze_mode == "Coming soon":
    st.markdown("Coming soon")




