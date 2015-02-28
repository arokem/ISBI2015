import numpy as np
import dipy.core.gradients as grad

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

def create_shells(): 
    # Dataset structure:
    # -----------------
    # 6 voxels a 3311 signals
    # in total 33 shells 
    # each shell consists of 2*45 antipodal DWI measurements
    # a quadrupel of 3 bvals has the same TE
    # this quadrupel hast 31 b0s

    # Read data
    data_dict = read_data()
    bvals = data_dict['seen']['bvals']
    bvecs = data_dict['seen']['bvecs']
    delta = data_dict['seen']['delta']
    Delta = data_dict['seen']['Delta']
    te = data_dict['seen']['TE']
    gtab = grad.gradient_table(bvals, bvecs, big_delta=Delta, small_delta=delta)

    # Categorise data into Shells and TE Quadrupel
    bvals_unique, shell  = np.unique(bvals, return_inverse=True)
    te_unique, te_unique_mask  = np.unique(te, return_inverse=True)

    # First entry for each quadrupel should be the b0s related to this TE
    shell_mask = {}
    for j in range(0, len(te_unique)):
        shell_mask[j] = {}
        shell_mask[j][0] = np.logical_and(te_unique_mask==j, gtab.b0s_mask)

    te_counter = np.ones(len(te_unique)).astype(int)
    for i in range(1, len(bvals_unique)): # for every shell
        for j in range(0, len(te_unique)): # look at TE's
            # and find the belonging one
            if np.any(np.logical_and(te_unique_mask==j, shell==i) == True):
                shell_mask[j][te_counter[j]] = shell==i
                te_counter[j] += 1
                
    return shell_mask

def split_data():
    mask = create_shells()

    # Pick random shells in TE quadrupel
    unseen_shells = np.random.random_integers(1,3, len(mask))

    # Create mask for unseen data
    mask_unseen = np.zeros(mask[0][0].shape).astype(bool)
    for i in range(0, len(mask)):
        mask_unseen = np.logical_or(mask_unseen, mask[i][unseen_shells[i]])

    mask_seen = np.invert(mask_unseen)
    
    return mask_seen, mask_unseen