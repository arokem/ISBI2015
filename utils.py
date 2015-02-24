import numpy as np

def b_value(g, delta, Delta, gamma=42.576):
    """ 
    Calculate the b value
    
    Parameters
    ----------
    g : gradient strength (mT/m, typically around 40)
    delta : gradient duration
    Delta : diffusion duration
    gamma : the gyromagnetic ration (42.576 MHz/T for Hydrogen)
    
    """
    G = g*1e-3*1e-6 #convert to T/um
    gamma = 2*np.pi*gamma*1e6*1e-3 # convert to 1/ms/T (Hz = cycles/sec, 
                                   # 1 cycle = 2pi = 2pi/sec) 
    b = gamma ** 2 * G ** 2 * delta ** 2 * (Delta-delta/3) # msec/um^2   
    return 1000 * b #s/mm^2

