import numpy as np
import dipy.reconst.sfm as sfm
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


def sfm_design_matrix(gtab, sphere, responses=my_responses):
        dm = []
        for response in responses :
            dm.append(sfm.sfm_design_matrix(gtab, sphere, response,
                                            'signal'))
        return np.concatenate(dm, -1)


def _kappa(zeta, n, l):
    return np.sqrt((2 * factorial(n - l)) / (zeta ** 1.5 * gamma(n + 1.5)))


def shore_design_matrix(radial_order, zeta, gtab, tau=1 / (4 * np.pi ** 2)):
    r"""Compute the SHORE matrix for modified Merlet's 3D-SHORE [1]_

    ..math::
            :nowrap:
                \begin{equation}
                    \textbf{E}(q\textbf{u})=\sum_{l=0, even}^{N_{max}}
                                            \sum_{n=l}^{(N_{max}+l)/2}
                                            \sum_{m=-l}^l c_{nlm}
                                            \phi_{nlm}(q\textbf{u})
                \end{equation}

    where $\phi_{nlm}$ is
    ..math::
            :nowrap:
                \begin{equation}
                    \phi_{nlm}^{SHORE}(q\textbf{u})=\Biggl[\dfrac{2(n-l)!}
                        {\zeta^{3/2} \Gamma(n+3/2)} \Biggr]^{1/2}
                        \Biggl(\dfrac{q^2}{\zeta}\Biggr)^{l/2}
                        exp\Biggl(\dfrac{-q^2}{2\zeta}\Biggr)
                        L^{l+1/2}_{n-l} \Biggl(\dfrac{q^2}{\zeta}\Biggr)
                        Y_l^m(\textbf{u}).
                \end{equation}

    Parameters
    ----------
    radial_order : unsigned int,
        an even integer that represent the order of the basis
    zeta : unsigned int,
        scale factor
    gtab : GradientTable,
        gradient directions and bvalues container class
    tau : float,
        diffusion time. By default the value that makes q=sqrt(b).

    References
    ----------
    .. [1] Merlet S. et. al, "Continuous diffusion signal, EAP and
    ODF estimation via Compressive Sensing in diffusion MRI", Medical
    Image Analysis, 2013.

    """
    mat_gtab = grad.gradient_table(gtab.bvals[~gtab.b0s_mask],
                                   gtab.bvecs[~gtab.b0s_mask])

    qvals = np.sqrt(mat_gtab.bvals / (4 * np.pi ** 2 * tau))
    qvals[mat_gtab.b0s_mask] = 0
    bvecs = mat_gtab.bvecs

    qgradients = qvals[:, None] * bvecs

    r, theta, phi = cart2sphere(qgradients[:, 0], qgradients[:, 1],
                                qgradients[:, 2])
    theta[np.isnan(theta)] = 0
    F = radial_order / 2
    n_c = np.round(1 / 6.0 * (F + 1) * (F + 2) * (4 * F + 3))
    M = np.zeros((r.shape[0], n_c))

    counter = 0
    for l in range(0, radial_order + 1, 2):
        for n in range(l, int((radial_order + l) / 2) + 1):
            for m in range(-l, l + 1):
                M[:, counter] = real_sph_harm(m, l, theta, phi) * \
                    genlaguerre(n - l, l + 0.5)(r ** 2 / zeta) * \
                    np.exp(- r ** 2 / (2.0 * zeta)) * \
                    _kappa(zeta, n, l) * \
                    (r ** 2 / zeta) ** (l / 2)
                counter += 1
    return M


class Model(sfm.SparseFascicleModel):
    @sfm.auto_attr
    def design_matrix(self):
        return sfm_design_matrix(self.gtab, self.sphere,
                                 responses=my_responses)
        #return shore_design_matrix(40, 2000, self.gtab)
        
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
        if gtab is None:
            _matrix = self.model.design_matrix
            gtab = self.model.gtab
        # The only thing we can't change at this point is the sphere we use
        # (which sets the width of our design matrix):
        else:
            _matrix = sfm_design_matrix(gtab, self.model.sphere, responses)
            #_matrix = shore_design_matrix(40, 2000, gtab)
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

