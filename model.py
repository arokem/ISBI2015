import numpy as np
import dipy.reconst.sfm as sfm

my_responses =[[0.0015, 0.0005, 0.0005],
               [0.001, 0.0005, 0.0005],
               [0.0015, 0.0003, 0.0003]]

def model_design_matrix(gtab, sphere, responses=my_responses):
        dm = []
        for response in responses :
            dm.append(sfm.sfm_design_matrix(gtab, sphere, response,
                                            'signal'))
        return np.concatenate(dm, -1)


class Model(sfm.SparseFascicleModel):
    @sfm.auto_attr
    def design_matrix(self):
        return model_design_matrix(self.gtab, self.sphere,
                                   responses=my_responses)
        
    def fit(self, data, mask=None):
        """
        Fit the SparseFascicleModel object to data

        Parameters
        ----------
        data : array
            The measured signal.

        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed. Has the shape `data.shape[:-1]`. Default: None,
            which implies that all points should be analyzed.

        Returns
        -------
        SparseFascicleFit object

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
            _matrix = model_design_matrix(gtab, self.model.sphere, responses)
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
