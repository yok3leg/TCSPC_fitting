import pandas as pd
import streamlit as st
import numpy as np

data_df = pd.DataFrame(
    {
        "favorite": [True, False, False, True],
        "widgets": ["st.selectbox", "st.number_input", "st.text_area", "st.button"],
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