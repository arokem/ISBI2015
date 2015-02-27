import numpy as np
import dipy.reconst.sfm as sfm
import dipy.reconst.dti as dti
from dipy.sims import voxel as sims
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sph_harm
from scipy.special import genlaguerre, gamma
from math import factorial
import dipy.core.gradients as grad
import scipy.optimize as opt
import scipy.stats as stats

my_responses = []
for ad in np.arange(0.001, 0.0018, 0.0005):
    for rd in np.arange(0, ad, 0.0005):
        this_response = [ad, rd, rd]
        my_responses.append(this_response)


class BiExponentialIsotropicModel(sfm.IsotropicModel):
    def _nlls_err_func(self, parameter, data):
        bvals = self.gtab.bvals[~self.gtab.b0s_mask]
        n = bvals.shape[0]
        noise = parameter[0]
        alpha_1 = parameter[1]
        alpha_2 = parameter[2]
        tensor_1 = parameter[3]
        tensor_2 = parameter[4]
        
        y = (noise +
             alpha_1 * np.exp(-bvals*tensor_1) +
             alpha_2 * np.exp(-bvals*tensor_2))
        residuals = data - y
        return residuals
    
    def fit(self, data):
        if len(data.shape) == 1:
            n_vox = 1
        else:
            n_vox = data.shape[0]

        data_no_b0 = data[..., ~self.gtab.b0s_mask]
        if np.sum(self.gtab.b0s_mask) > 0:
            s0 = np.mean(data[..., self.gtab.b0s_mask], -1)
            to_fit = data_no_b0 / s0[..., None]
        else:
            to_fit = data_no_b0
    
        start_params = np.ones((n_vox, 5))
        start_params[:,0] *= 0.0
        start_params[:,1] *= 0.8
        start_params[:,2] *= 0.2
        start_params[:,3] *= 0.001
        start_params[:,4] *= 0.0002
        
        params, status = opt.leastsq(self._nlls_err_func,
                                     start_params,
                                     args=(to_fit.squeeze()))
        
        return BiExponentialIsotropicFit(self, params, n_vox)


class BiExponentialIsotropicFit(sfm.IsotropicFit):
    def predict(self, gtab=None):
        if gtab is None:
            gtab = self.model.gtab
            
        bvals = gtab.bvals[~gtab.b0s_mask]    
        noise = self.params[0]
        alpha_1 = self.params[1]
        alpha_2 = self.params[2]
        tensor_1 = self.params[3]
        tensor_2 = self.params[4]    
            
        y = (noise +
             alpha_1*np.exp(-bvals*tensor_1) +
             alpha_2*np.exp(-bvals*tensor_2))    
        return y


def response_design_matrix(gtab, sphere, response, p_ad=None, p_rd=None):
    """
    A response-function dependent design matrix
    """
    mat_gtab = grad.gradient_table(gtab.bvals[~gtab.b0s_mask],
                                       gtab.bvecs[~gtab.b0s_mask])
    mat = np.empty((np.sum(~gtab.b0s_mask),
                        sphere.vertices.shape[0]))

    # Calculate column-wise:
    for ii, this_column in enumerate(sphere.vertices):
        # Rotate the canonical tensor towards this vertex and calculate the
        # signal you would have gotten in the direction
        evecs = sims.all_tensor_evecs(this_column)
        if p_ad is not None and p_rd is not None:
            # And if you have p_ad and p_rd then also row-wise:
            for jj, row_bvec in enumerate(mat_gtab.bvecs):
                row_gtab = grad.gradient_table([mat_gtab.bvals[jj]],
                                               [row_bvec])
                this_ad = np.exp(np.polyval(p_ad, row_gtab.bvals[0]))
                this_rd = np.exp(np.polyval(p_rd, row_gtab.bvals[0]))
                response = [this_ad, this_rd, this_rd]
                sig = sims.single_tensor(row_gtab, evals=response,
                                             evecs=evecs)
                # For regressors based on the single tensor, remove $e^{-bD}$
                iso_sig = np.exp(-row_gtab.bvals * np.mean(response))
                mat[jj, ii] = sig - iso_sig
        else:
            sig = sims.single_tensor(mat_gtab, evals=response,
                                     evecs=evecs)
            iso_sig = np.exp(-mat_gtab.bvals * np.mean(response))
            mat[:, ii] = sig - iso_sig

    return mat


def sfm_design_matrix(gtab, sphere, responses=my_responses, p_ad=None,
                                                            p_rd=None):
    """
    This creates the full design matrix out of individual responses functions
    and from the fit across
    """
    dm = []
    for response in responses :
        dm.append(response_design_matrix(gtab, sphere, response))

    if p_ad is not None and p_rd is not None:
        dm.append(response_design_matrix(gtab, sphere, response, p_ad, p_rd))
    return np.concatenate(dm, -1)


class Model(sfm.SparseFascicleModel):
    @sfm.auto_attr
    def design_matrix(self):
        return sfm_design_matrix(self.gtab, self.sphere,
                                 responses=my_responses, p_ad=self.p_ad,
                                 p_rd=self.p_rd)
        
    def fit(self, data, TE=None, te_order=3, mask=None):
        """
        Fit the SparseFascicleModel object to data

        Parameters
        ----------
        data : array
            The measured signal.

        TE : the measurement TE.

        te_order : degree of the polynomial to fit to the TE-dependence of S0
        
        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed. Has the shape `data.shape[:-1]`. Default: None,
            which implies that all points should be analyzed.

        Returns
        -------
        Fit object

        """
        te_params = np.polyfit(TE[..., self.gtab.b0s_mask],
                               np.log(data[..., self.gtab.b0s_mask]), te_order)

        data_no_te = np.zeros_like(data)
        for ii in range(data_no_te.shape[0]):
            this_te = TE[ii]
            te_idx = (TE==this_te)
            this_s0 = data[te_idx * self.gtab.b0s_mask]
            # Make sure that signals are no higher than their relevant S0 (the
            # ones with the corresponding TE). If they are, correct them:
            #if data[ii] > stats.scoreatpercentile(this_s0, 86):
            #    print("this!")
            #    data[ii] = stats.scoreatpercentile(this_s0, 86)
            te_est = np.exp(np.polyval(te_params, this_te))
            data_no_te[ii] = data[ii] / te_est

        AD = []
        RD = []
        bvals = []
        for this_bval in np.unique(self.gtab.bvals[~self.gtab.b0s_mask]):
            bval_idx = (self.gtab.bvals==this_bval)
            this_gtab = grad.gradient_table(
                np.hstack([self.gtab.bvals[self.gtab.b0s_mask],
                           self.gtab.bvals[bval_idx]]),
                np.vstack([self.gtab.bvecs[self.gtab.b0s_mask],
                           self.gtab.bvecs[bval_idx]]))
            
            tensor_model = dti.TensorModel(this_gtab)
            tensor_fit = tensor_model.fit(
                np.hstack([data_no_te[self.gtab.b0s_mask],
                           data_no_te[bval_idx]]))

            bvals.append(this_bval)
            AD.append(tensor_fit.ad)
            RD.append(tensor_fit.rd)

        self.p_ad = np.polyfit(bvals, np.log(AD), 2)
        self.p_rd = np.polyfit(bvals, np.log(RD), 2)

        sf_fit = sfm.SparseFascicleModel.fit(self, data_no_te, mask)
        return Fit(sf_fit, te_params)
        
        
class Fit(sfm.SparseFascicleFit):
    def __init__(self, sf_fit, te_params):
        self.model = sf_fit.model
        self.beta = sf_fit.beta
        self.S0 = sf_fit.S0
        self.iso = sf_fit.iso
        self.te_params = te_params

        
    def predict(self, gtab, TE, responses=my_responses, S0=None):
        """
        Predict the signal based on the SFM parameters

        Parameters
        ----------
        gtab : GradientTable, optional
            The bvecs/bvals to predict the signal on. Default: the gtab from
            the model object.
        response : list of 3 elements, optional
            The eigenvalues of a tensor which will serve as a kernel
            function. Default: the response of the model object. Default to use
            `model.response`.
        S0 : float or array, optional
             The non-diffusion-weighted signal. Default: use the S0 of the data

        Returns
        -------
        pred_sig : ndarray
            The signal predicted in each voxel/direction
        """
        _matrix = sfm_design_matrix(gtab, self.model.sphere, responses,
                                    self.model.p_ad, self.model.p_rd)
        # Get them all at once:
        beta_all = self.beta.reshape(-1, self.beta.shape[-1])
        pred_weighted = np.dot(_matrix, beta_all.T).T
        pred_weighted = pred_weighted.reshape(self.beta.shape[:-1] +
                                              (_matrix.shape[0],))

        if S0 is None:
            S0 = self.S0
        if isinstance(S0, np.ndarray):
            S0 = S0[..., None]

        iso_signal = self.iso.predict(gtab)

        pre_pred_sig = S0 * (pred_weighted +
                             iso_signal.reshape(pred_weighted.shape))
        pred_sig = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))
        pred_sig[..., ~gtab.b0s_mask] = pre_pred_sig
        pred_sig[..., gtab.b0s_mask] = S0

        predict_with_te = np.zeros_like(pred_sig)
        for ii in range(predict_with_te.shape[0]):
            this_te = TE[ii]
            te_idx = (TE==this_te)
            te_est = np.exp(np.polyval(self.te_params, this_te))
            predict_with_te[ii] = pred_sig[ii] * te_est

        return predict_with_te


