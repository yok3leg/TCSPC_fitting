import numpy as np
from lmfit import Model, Parameters

def irf_convolution(params, t, irf, decay_func):
    """Convolve the IRF with the decay function."""
    decay = decay_func(t, params)
    return np.convolve(decay, irf, mode='full')[:len(t)]

def decay_model(t, params):
    """Example single exponential decay model."""
    amplitude = params['amplitude']
    lifetime = params['lifetime']
    return amplitude * np.exp(-t / lifetime)

def reconvolution_fit(t, decay_data, irf, initial_params):
    """
    Perform reconvolution fit for fluorescence decay data using an IRF.
    
    Parameters:
    - t: Time array (1D numpy array).
    - decay_data: Fluorescence decay data (1D numpy array).
    - irf: Instrument Response Function (1D numpy array).
    - initial_params: Initial guesses for the parameters (lmfit.Parameters object).
    
    Returns:
    - result: Fitting result (lmfit.ModelResult object).
    """

    # Define a custom model for the fit using the convolution
    def model_func(t, amplitude, lifetime):
        params = Parameters()
        params.add('amplitude', value=amplitude)
        params.add('lifetime', value=lifetime)
        return irf_convolution(params, t, irf, decay_model)
    
    # Create a model instance
    model = Model(model_func, independent_vars=['t'])

    # Perform the fit
    result = model.fit(decay_data, t=t, amplitude=initial_params['amplitude'], lifetime=initial_params['lifetime'])

    return result