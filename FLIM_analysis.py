import numpy as np
import lmfit

def single_exponential(t, A, tau, C):
    """
    Single exponential decay function.
    
    t: time
    A: amplitude
    tau: lifetime
    C: baseline offset
    """
    return A * np.exp(-t / tau) + C

def calculate_lifetimes_lmfit(data, time_bins):
    """
    Calculate the fluorescence lifetime for each pixel in a 2D array using lmfit.
    
    data: 2D numpy array where rows are pixels and columns are time bins.
    time_bins: 1D numpy array or list representing the time values for each bin.
    
    Returns:
    lifetimes: 1D numpy array of fluorescence lifetimes for each pixel.
    """
    num_pixels = data.shape[0]
    lifetimes = np.zeros(num_pixels)

    model = lmfit.Model(single_exponential)

    for i in range(num_pixels):
        y_data = data[i, :]
        params = model.make_params(A=np.max(y_data), tau=(time_bins[-1] - time_bins[0]) / 2, C=np.min(y_data))
        
        try:
            # Perform the fit
            result = model.fit(y_data, params, t=time_bins)
            # Extract the lifetime (tau) from the fitting result
            lifetimes[i] = result.params['tau'].value
        except Exception as e:
            # Handle the case where the fit fails
            lifetimes[i] = np.nan
    
    return lifetimes