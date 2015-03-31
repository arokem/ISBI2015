
# coding: utf-8

# In[1]:

import numpy as np
#%matplotlib inline
import dipy.reconst.sfm as sfm
import dipy.reconst.csdeconv as csd
import dipy.reconst.dti as dti
import dipy.core.gradients as grad
import dipy.reconst.cross_validation as xval
import dipy.data as dpd
import utils
import model as mm 
from model import Model, BiExponentialIsotropicModel, my_responses
from parallelization import parallelization
import csv
import sklearn.linear_model as lm


# In[2]:

version = 'final3'


# In[3]:

# Load Data
data_dict = utils.read_data()
data = data_dict['seen']['signal']
dataX = data_dict['seen']['signalX']
bvals_seen = data_dict['seen']['bvals']
bvecs_seen = data_dict['seen']['bvecs']
delta_seen = data_dict['seen']['delta']
Delta_seen = data_dict['seen']['Delta']
te_seen = data_dict['seen']['TE']
g_seen = data_dict['seen']['g']
bvals_unseen = data_dict['unseen']['bvals']
bvecs_unseen = data_dict['unseen']['bvecs']
delta_unseen = data_dict['unseen']['delta']
Delta_unseen = data_dict['unseen']['Delta']
te_unseen = data_dict['unseen']['TE']
g_unseen = data_dict['unseen']['g']
gtab_seen= grad.gradient_table(bvals_seen, bvecs_seen, big_delta=Delta_seen, small_delta=delta_seen)
gtab_unseen= grad.gradient_table(bvals_unseen, bvecs_unseen, big_delta=Delta_unseen, small_delta=delta_unseen)


# In[4]:

data_seen_list = []
data_unseen_list = []
te_seen_list = []
te_unseen_list = []
gtab_seen_list = []
gtab_unseen_list = []
g_seen_list = []
g_unseen_list = []

for d in [data, dataX]:
    for m in range(data.shape[-1]): # for every voxel
        data_seen_list.append(d[:,m])
        te_seen_list.append(te_seen)
        te_unseen_list.append(te_unseen)
        g_seen_list.append(g_seen)
        g_unseen_list.append(g_unseen)
        gtab_seen_list.append(gtab_seen)
        gtab_unseen_list.append(gtab_unseen)


# In[5]:

alphas = [5e-7]
l1_ratios = [0.4]


# In[7]:

n_variations = len(alphas) * len(l1_ratios)
n_voxels = len(data_seen_list)
n = n_voxels * n_variations


# In[8]:

alpha_grid, l1_ratio_grid = np.meshgrid(alphas, l1_ratios)
alpha_grid = np.reshape(alpha_grid, (n_variations, -1)).squeeze()
l1_ratio_grid = np.reshape(l1_ratio_grid, (n_variations, -1)).squeeze()


# In[9]:

alpha = np.repeat(alpha_grid, n_voxels).tolist()
l1_ratio = np.repeat(l1_ratio_grid, n_voxels).tolist()


# In[10]:

solvers = []
for a, l in zip(alpha, l1_ratio):
    solvers.append(lm.ElasticNet(l1_ratio=l, alpha=a, positive=True, warm_start=True, max_iter=25000, fit_intercept=True, normalize=True))


# In[11]:

p = parallelization()


# In[13]:

models = p.start(Model, n, gtab_seen_list*n_variations, isotropic=[BiExponentialIsotropicModel], solver=solvers)


# In[12]:

fits = p.start([i.fit for i in models], n, data_seen_list*n_variations, te_seen_list*n_variations,  g_seen_list*n_variations)


# In[ ]:

predicts = p.start([i.predict for i in fits], n, gtab_unseen_list*n_variations, te_unseen_list*n_variations)


# In[ ]:

all_predictions = np.asarray(predicts)


# In[ ]:

np.savetxt('unseenSignal.txt', all_predictions[:6].T, fmt='%.4f', header='%—voxel1——-voxel2——-voxel3-—-voxel4——-voxel5——-voxel6')


# In[ ]:

np.savetxt('unseenSignalX.txt', all_predictions[6:].T, fmt='%.4f', header='%—voxel1——-voxel2——-voxel3-—-voxel4——-voxel5——-voxel6')


# In[32]:

filename = 'predictions'
with open(filename + '_' + version + '.csv', 'w') as f:
   writer = csv.writer(f, delimiter=',')
   writer.writerows(predicts)


# In[ ]:

betas = []
te_params = []
iso_params = []
S0s = []
for i in fits:
    betas.append(i.beta)
    te_params.append(i.te_params)
    iso_params.append(i.iso.params)
    S0s.append([i.S0])


# In[ ]:

filename = 'betas'
with open(filename + '_' + version + '.csv', 'w') as f:
   writer = csv.writer(f, delimiter=',')
   writer.writerows(betas)


# In[ ]:

filename = 'te_params'
with open(filename + '_' + version + '.csv', 'w') as f:
   writer = csv.writer(f, delimiter=',')
   writer.writerows(te_params)


# In[ ]:

filename = 'iso_params'
with open(filename + '_' + version + '.csv', 'w') as f:
   writer = csv.writer(f, delimiter=',')
   writer.writerows(iso_params)


# In[ ]:

filename = 'S0s'
with open(filename + '_' + version + '.csv', 'w') as f:
   writer = csv.writer(f, delimiter=',')
   writer.writerows(S0s)

