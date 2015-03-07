import numpy as np
import dipy.core.gradients as grad

import sys, time
try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

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

class ProgressBar:
    def __init__(self, iterations, msg = None):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.msg = msg
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        try:
            clear_output()
        except Exception:
            # terminal IPython has no clear_output
            pass
        print '\r', self,
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = self.msg + '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) / 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)