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

def fit_LifeData(fluor_life, tau0):
    fluor_life.reconvolution_fit(tau0, verbose=False)    
    return fluor_life

def get_LifeFitparams(fluor_life, n_decays):
    fit_parameters = pd.DataFrame(fluor_life.fit_param, columns=['tau', 'ampl'])
    fit_parameters['tau'] = ['{:0.2f} +/- {:0.2f}'.format(val,err) for val, err in zip(fluor_life.fit_param['tau'], fluor_life.fit_param_std['tau'])]
    fit_parameters['ampl'] = ['{:0.2f}'.format(val) for val in fluor_life.fit_param['ampl']]
    fit_parameters.index = ['tau{:d}'.format(i+1) for i in range(n_decays)]
    fit_parameters = fit_parameters._append(pd.DataFrame({'tau':'{:0.2f} +/- {:0.2f}'.format(fluor_life.av_lifetime,fluor_life.av_lifetime_std), 'ampl':'-'}, index=['weighted tau']))
    fit_parameters.columns = ['lifetime (ns)', 'weight']
    return fit_parameters

def to_base64(df):
    csv = df.to_csv(index=False, float_format='%.3f')
    return base64.b64encode(csv.encode()).decode()

def main():
    st.title('TCSPC fitting')
    st.sidebar.image('https://github.com/yok3leg/TCSPC_fitting/blob/main/logo.png?raw=true', width=300)
    st.markdown('This software is inspired by https://github.com/fdsteffen/Lifefit/ (https://pubs.rsc.org/en/content/articlelanding/2016/CP/C6CP04277E)') 
    lifetime()

def lifetime():
    st.info('&rarr; Documents and DEMO data can be found here: https://github.com/yok3leg/TCSPC_fitting')
    fluor_buffer = st.file_uploader('Fluorescence lifetime decay', 'txt')
    if fluor_buffer is not None:
        fluor, timestep_ns = lf.tcspc.read_decay(io.TextIOWrapper(fluor_buffer), 'Horiba')
        if fluor is not None:
            irf_type = st.radio(label='IRF', options=('Gaussian IRF', 'experimental IRF'), index=0)
            if irf_type == 'experimental IRF':
                irf_buffer = st.file_uploader('IRF decay', 'txt')
                if irf_buffer is not None:
                    irf, _ = lf.tcspc.read_decay(io.TextIOWrapper(irf_buffer))
                    if irf is not None:
                        gauss_sigma = None
                    else:
                        irf = False
            else:
                irf = None
                irf_buffer = False
                gauss_sigma = st.number_input('IRF sigma', min_value=0.00, value=0.10, step=0.01, format='%0.2f')
        st.write('---')

    if (fluor_buffer is not None) and (irf_buffer is not None): 
        if (fluor is not None) and (irf is not False):

            st.sidebar.markdown('### Parameters for reconvolution')
            n_decays = st.sidebar.number_input('number of exponential decays', value=2, min_value=1, max_value=4, step=1)
            tau0 = []
            #col = st.sidebar.columns(n_decays)
            for i in range(n_decays):
                tau0.append(st.sidebar.number_input('tau{:d}'.format(i+1), value=float(10**i), step=float(10**(i-1)), format='%0.{prec}f'.format(prec=max(1-i, 0))))

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