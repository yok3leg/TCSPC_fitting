import streamlit as st
import lifefit as lf
import numpy as np
import pandas as pd
import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
import time
import base64
import scipy.signal as sig

def fit_LifeData(fluor_life, tau0, irf_shift = 0):
    fluor_life.reconvolution_fit(tau0, verbose=False, irf_shift = irf_shift)    
    return fluor_life

def get_LifeFitparams(fluor_life, n_decays):
    fit_parameters = pd.DataFrame(fluor_life.fit_param, columns=['tau', 'ampl'])
    fit_parameters['tau'] = ['{:0.3f} +/- {:0.3f}'.format(val,err) for val, err in zip(fluor_life.fit_param['tau'], fluor_life.fit_param_std['tau'])]
    fit_parameters['ampl'] = ['{:0.3f}'.format(val) for val in fluor_life.fit_param['ampl']]
    #st.write(fluor_life.fit_param) # for debug
    fit_parameters.index = ['tau{:d}'.format(i+1) for i in range(n_decays)]
    fit_parameters = fit_parameters._append(pd.DataFrame({'tau':'{:0.2f} +/- {:0.2f}'.format(fluor_life.av_lifetime,fluor_life.av_lifetime_std), 'ampl':'-'}, index=['weighted tau']))
    fit_parameters.columns = ['lifetime (ns)', 'weight']
    return fit_parameters

def to_base64(df):
    csv = df.to_csv(index=False, float_format='%.3f')
    return base64.b64encode(csv.encode()).decode()

def shift_hist(input,n):
    output = np.zeros(input.shape[0])
    if n < 0: # left
        output[0:input.shape[0]+n] = input[-n-1:-1]
    elif n > 0: # right
        output[n:-1] = input[0:input.shape[0]-n-1]
    else:
        output = input
    return output        

def bin_1darray(c,n):
    # for 1d array
    c2 = np.reshape(c,(int(len(c)//n),n))
    c3 = np.sum(c2,axis=1)
    return c3

def main():
    st.title('TCSPC fitting')
    st.sidebar.image('https://github.com/yok3leg/TCSPC_fitting/blob/main/logo.png?raw=true', width=300)
    st.markdown('This software is inspired by https://github.com/fdsteffen/Lifefit/ (https://pubs.rsc.org/en/content/articlelanding/2016/CP/C6CP04277E)') 
    lifetime()

def lifetime():
    st.info('&rarr; Documents and DEMO data can be found here: https://github.com/yok3leg/TCSPC_fitting')
    fluor_buffer = st.file_uploader('Fluorescence lifetime decay', 'txt')
    if fluor_buffer is not None:
        reverse_mode = st.checkbox("Reverse mode")
        fluor, timestep_ns = lf.tcspc.read_decay(io.TextIOWrapper(fluor_buffer), 'Horiba')
        if reverse_mode:
            fluor[:,1] = np.flipud(fluor[:,1])
            bg = np.mean(fluor[0:200,1])
            fluor[:,1] = sig.savgol_filter(fluor[:,1],5,3)
            fluor[:,1] =  fluor[:,1] - bg
            # fluor = fluor[0:800,:]  
        if fluor is not None:
            irf_type = st.radio(label='IRF', options=('Gaussian IRF', 'experimental IRF'), index=0)
            if irf_type == 'experimental IRF':
                irf_buffer = st.file_uploader('IRF decay', 'txt')
                if irf_buffer is not None:
                    irf, _ = lf.tcspc.read_decay(io.TextIOWrapper(irf_buffer))
                    if reverse_mode:
                        irf[:,1] = np.flipud(irf[:,1])
                        bg = bg = np.mean(irf[0:200,1])
                        irf[:,1] = sig.savgol_filter(irf[:,1],5,3)
                        irf[:,1] = irf[:,1] - bg*2
                        irf[:,1] = irf[:,1]*(np.max(fluor[:,1])/np.max(irf[:,1]))
                        # irf = irf[0:800,:]  
                    fluor_peak = np.argmax(fluor[:,1])
                    irf_peak = np.argmax(irf[:,1])
                    shift_factor = irf_peak-fluor_peak
                    st.write(shift_factor)
                    # fluor[:,1] = shift_hist(fluor[:,1],shift_factor)
                    if irf is not None:
                        gauss_sigma = None
                    else:
                        irf = False
            else:
                irf = None 
                irf_buffer = False
                st.latex(r'''I_{IRF} = I_0\exp{(\frac{-(t-t_0)^2}{2\sigma^2})}''')
                gauss_sigma = st.number_input('IRF sigma', min_value=0.000, value=0.100, step=0.001, format='%0.3f')
        st.write('---')

    if (fluor_buffer is not None) and (irf_buffer is not None): 
        if (fluor is not None) and (irf is not False):

            st.sidebar.markdown('### Parameters for reconvolution')
            n_decays = st.sidebar.number_input('number of exponential decays', value=2, min_value=1, max_value=4, step=1)
            tau0 = []
            #col = st.sidebar.columns(n_decays)
            for i in range(n_decays):
                tau0.append(st.sidebar.number_input('tau{:d}'.format(i+1), value=float(10**(i-1)+0.1), step=float(10**(i-1)), format='%0.{prec}f'.format(prec=max(1-i, 0))))

            change_area = st.checkbox('Change fitting area (Experimental)')
            if change_area == True:
                xlimits = st.slider('Select a time limits (ns) on the x-axis', float(1*timestep_ns), float(len(fluor)*timestep_ns), (float(1*timestep_ns), float(len(fluor)*timestep_ns)), format='%0.01f ns')
                bin_start = round(xlimits[0]/timestep_ns)
                bin_stop = round(xlimits[1]/timestep_ns)
                print(bin_start)
                print(bin_stop)
                fluor_life = lf.tcspc.Lifetime(fluor[bin_start:bin_stop], timestep_ns, irf, gauss_sigma=gauss_sigma)
            else:
                fluor_life = lf.tcspc.Lifetime(fluor, timestep_ns, irf, gauss_sigma=gauss_sigma)
            
            with st.spinner('Fitting...'):
                try:
                    fluor_life = fit_LifeData(fluor_life, tau0)
                except:
                    st.warning('Fit did not converge.')
                else:
                    st.success('Reconvolution fit successful!')

                    fit_parameters = get_LifeFitparams(fluor_life, n_decays)
                    st.table(fit_parameters)
                    # st.write(fit_parameters)

                    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1)
                    xlimits = st.slider('Select a time limits (ns) on the x-axis (does not affect the fit)', min(fluor_life.fluor[fluor_life.fluor[:,2]>0,0]), max(fluor_life.fluor[fluor_life.fluor[:,2]>0,0]), (20.0, 80.0), format='%0.1f ns')
                    fig.add_trace(go.Scatter(x = fluor_life.irf[:,0], y = fluor_life.irf[:,2], line=dict(color='rgb(200, 200, 200)', width=1), name='IRF'), row=1, col=1)
                    fig.add_trace(go.Scatter(x = fluor_life.fluor[:,0], y = fluor_life.fluor[:,2], line=dict(color='rgb(79, 115, 143)', width=1), name='data'), row=1, col=1)
                    fig.add_trace(go.Scatter(x = fluor_life.fluor[:,0], y = fluor_life.fit_y, line=dict(color='black', width=1), name='fit'), row=1, col=1)
                    fig.add_trace(go.Scatter(x = fluor_life.fluor[:,0], y = fluor_life.fluor[:,2]-fluor_life.fit_y, line=dict(color='rgb(79, 115, 143)', width=1), name='fit'), row=2, col=1)
                    fig.update_layout(yaxis_type="log", template='none', xaxis2_title='time (ns)', yaxis_title='counts', yaxis2_title='residuals', xaxis1 = dict(range = xlimits), xaxis2 = dict(range = xlimits), yaxis1 = dict(range = (0,4)), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    data, parameters = fluor_life._serialize()
                    data_df = pd.DataFrame(data)
                    data_df = data_df.loc[(data_df.time>=xlimits[0]) & (data_df.time<= xlimits[1])]
                    st.write('### Export the data and fit parameters')
                    if st.checkbox('Show TCSPC data (json)'):
                        st.write('**Note:** The entire json formatted TCSPC dataset can be copied to the clipboard by clicking on the topmost blue chart icon.')
                        st.json(data)
                    if st.checkbox('Show TCSPC data (table)'):
                        b64 = to_base64(data_df)
                        href = f'<a href="data:file/csv;base64,{b64}" download="lifetime.csv">Download as .csv</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        st.table(data_df)
                    if st.checkbox('Show fit parameters (json)'):
                        st.write('**Note:** The json formatted fit parameters can be copied to the clipboard by clicking on the topmost blue chart icon.')  
                        st.json(parameters)
        else:
            st.error('File has a wrong format.')

if __name__ == "__main__":
    main()