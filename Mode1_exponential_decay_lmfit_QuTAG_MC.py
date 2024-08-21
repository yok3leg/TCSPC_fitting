import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

# Define n-exponential decay function
def n_exponential_decay(t, **params): #TODO use lmfit model
    n = int(len(params) / 2)
    decay = np.zeros_like(t)
    for i in range(n):
        A = params[f'A{i+1}']
        tau = params[f'tau{i+1}']
        decay += A * np.exp(-t / tau)
    return decay

def fitting(n_components,y_data,t):
    model = Model(n_exponential_decay) # Create the model
    params = Parameters() # Create Parameters object for initial guesses
    for i in range(n_components):
        params.add(f'A{i+1}', value=1.0, min=0)  # Amplitude
        params.add(f'tau{i+1}', value=1.0, min=0)  # lifetime
    result = model.fit(y_data, params=params, t=t) # Fit the model to the data
    return result, result.best_fit

