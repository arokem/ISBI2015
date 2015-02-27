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


def read_data():
    
    data = {}
    for this in ['seen', 'unseen']:
        scheme = np.loadtxt('./%sScheme.txt'%this, skiprows=1)
        data[this] = {}
        data[this]['bvecs'] = scheme[:, :3]    
        data[this]['g'] = scheme[:, 3] * 1000
        data[this]['Delta'] = scheme[:, 4] * 1000
        data[this]['delta'] = scheme[:, 5] * 1000
        data[this]['TE'] = scheme[:, 6] * 1000
        data[this]['bvals'] = b_value(data[this]['g'], data[this]['delta'],
                                     data[this]['Delta'])
    data['seen']['signal'] = np.loadtxt('./seenSignal.txt', skiprows=1)
    data['seen']['signalX'] =  np.loadtxt('./seenSignaX.txt', skiprows=1)

    return data

def ADC(data, gtab, TE):

    norm_sig = np.zeros_like(data)
    for ii in range(norm_sig.shape[0]):
            this_te = TE[ii]
            te_idx = (TE==this_te)
            te_s0 = np.mean(data[te_idx * gtab.b0s_mask])
            norm_sig[ii] = data[ii] / te_s0
    
    return -np.log(norm_sig) / gtab.bvals


def LSE(prediction, signal, sigma=8):
    return np.mean(((prediction - np.sqrt(signal**2 + sigma**2))**2)/(sigma**2))


def RMSE(prediction, signal):
    return np.sqrt(np.mean((prediction - signal)**2))





