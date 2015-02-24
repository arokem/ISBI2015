import numpy as np
import dipy.reconst.sfm as sfm
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sph_harm
from scipy.special import genlaguerre, gamma
from math import factorial
import dipy.core.gradients as grad

my_responses =[[0.0015, 0.0005, 0.0005],
               [0.001, 0.0005, 0.0005],
               [0.0015, 0.0003, 0.0003],
               [0.002, 0.001, 0.001]
               ]

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
        #return sfm_design_matrix(self.gtab, self.sphere,
        #                         responses=my_responses)
        return shore_design_matrix(40, 2000, self.gtab)
        
    def fit(self, data, mask=None):
        """
        Fit the SparseFascicleModel object to data

        Parameters
        ----------
        data : array
            The measured signal.

        TE : the measurement TE.
        
        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed. Has the shape `data.shape[:-1]`. Default: None,
            which implies that all points should be analyzed.

        Returns
        -------
        Fit object

        """
        if mask is None:
            flat_data = np.reshape(data, (-1, data.shape[-1]))
        else:
            mask = np.array(mask, dtype=bool, copy=False)
            flat_data = np.reshape(data[mask], (-1, data.shape[-1]))

        # Fitting is done on the relative signal (S/S0):
        flat_S0 = np.mean(flat_data[..., self.gtab.b0s_mask], -1)
        flat_S = flat_data[..., ~self.gtab.b0s_mask] / flat_S0[..., None]
        flat_isotropic = self.isotropic(self.gtab).fit(flat_data).predict()
        flat_params = np.zeros((flat_data.shape[0],
                                self.design_matrix.shape[-1]))

        for vox, vox_data in enumerate(flat_S):
            
            if np.any(np.isnan(vox_data)):
                pass
            else:
                fit_it = vox_data - flat_isotropic[vox]
                flat_params[vox] = self.solver.fit(self.design_matrix,
                                                   fit_it).coef_
        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            beta = flat_params.reshape(out_shape)
            iso_out = flat_isotropic.reshape(out_shape)
            S0 = flat_S0.reshape(out_shape).squeeze()
        else:
            beta = np.zeros(data.shape[:-1] +
                            (self.design_matrix.shape[-1],))
            beta[mask, :] = flat_params
            iso_out = np.zeros(data[..., ~self.gtab.b0s_mask].shape)
            iso_out[mask, ...] = flat_isotropic.squeeze()
            S0 = np.zeros(data.shape[:-1])
            S0[mask] = flat_S0

        return Fit(self, beta, S0, iso_out.squeeze())

class Fit(sfm.SparseFascicleFit):
    def predict(self, gtab=None, responses=my_responses, S0=None):
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
            #_matrix = sfm_design_matrix(gtab, self.model.sphere, responses)
            _matrix = shore_design_matrix(40, 2000, gtab)
        # Get them all at once:
        beta_all = self.beta.reshape(-1, self.beta.shape[-1])
        pred_weighted = np.dot(_matrix, beta_all.T).T
        pred_weighted = pred_weighted.reshape(self.beta.shape[:-1] +
                                              (_matrix.shape[0],))
        if S0 is None:
            S0 = self.S0
        if isinstance(S0, np.ndarray):
            S0 = S0[..., None]
        if isinstance(self.iso, np.ndarray):
            iso_signal = self.iso[..., None]
        pre_pred_sig = S0 * (pred_weighted + iso_signal.squeeze())
        pred_sig = np.zeros(pre_pred_sig.shape[:-1] + (gtab.bvals.shape[0],))
        pred_sig[..., ~gtab.b0s_mask] = pre_pred_sig
        pred_sig[..., gtab.b0s_mask] = S0
        return pred_sig.squeeze()
