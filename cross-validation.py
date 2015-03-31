
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set()
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

version = '7_3'


# In[3]:

sphere = dpd.get_sphere()
#sphere = sphere.subdivide(1)


# In[4]:

# Load Data
data_dict = utils.read_data()
data = data_dict['seen']['signal'][:, range(1)]
bvals = data_dict['seen']['bvals']
bvecs = data_dict['seen']['bvecs']
delta = data_dict['seen']['delta']
Delta = data_dict['seen']['Delta']
te = data_dict['seen']['TE']
g = data_dict['seen']['g']


# In[8]:

data_seen = []
data_unseen = []
te_seen = []
te_unseen = []
gtab_seen = []
gtab_unseen = []
g_seen = []
g_unseen = []

mask = utils.create_shells()
for m in range(data.shape[-1]): # for every voxel
    for j in range(0, len(mask)): # for every TE
        for i in range(0, len(mask[j])): # for every shell
            mask_unseen = mask[j][i]
            mask_seen = np.invert(mask_unseen)
            data_seen.append(data[mask_seen, m])
            data_unseen.append(data[mask_unseen, m])
            te_seen.append(te[mask_seen])
            te_unseen.append(te[mask_unseen])
            g_unseen.append(g[mask_unseen])
            g_seen.append(g[mask_seen])
            bvals_seen_temp = bvals[mask_seen]
            bvecs_seen_temp = bvecs[mask_seen]
            delta_seen_temp = delta[mask_seen]
            Delta_seen_temp = Delta[mask_seen]
            gtab_seen.append(grad.gradient_table(bvals_seen_temp, bvecs_seen_temp, big_delta=Delta_seen_temp, small_delta=delta_seen_temp))
            bvals_unseen_temp = bvals[mask_unseen]
            bvecs_unseen_temp = bvecs[mask_unseen]
            delta_unseen_temp = delta[mask_unseen]
            Delta_unseen_temp = Delta[mask_unseen]
            gtab_unseen.append(grad.gradient_table(bvals_unseen_temp, bvecs_unseen_temp, big_delta=Delta_unseen_temp, small_delta=delta_unseen_temp))


# In[6]:

# 3 shells, 11 TEs, 6 Voxel


# In[ ]:

'''
# Version 1
alphas = [1e-5]
l1_ratios = [0.1]

# Version 2
alphas = [1e-9, 1e-3]
l1_ratios = [0.3, 0.9]

# Version 3
alphas = [1e-12, 1e-5]
l1_ratios = [0.05, 0.3]

# Version 4
alphas = [5e-5, 1e-5, 5e-6]
l1_ratios = [0.25, 0.3, 0.35]

# Version 5
alphas = [5e-6, 5e-7, 5e-8, 5e-9]
l1_ratios = [0.35, 0.4, 0.45]

# Version 6
alphas = [1e-6, 5e-7, 1e-7]
l1_ratios = [0.375, 0.4, 0.425]
'''
# Version 7
alphas = [5e-7]
l1_ratios = [0.4]


# In[8]:

n_variations = len(alphas) * len(l1_ratios)
n_voxels = len(data_seen)
n = n_voxels * n_variations


# In[25]:

alpha_grid, l1_ratio_grid = np.meshgrid(alphas, l1_ratios)
alpha_grid = np.reshape(alpha_grid, (n_variations, -1)).squeeze()
l1_ratio_grid = np.reshape(l1_ratio_grid, (n_variations, -1)).squeeze()


# In[26]:

alpha = np.repeat(alpha_grid, n_voxels).tolist()
l1_ratio = np.repeat(l1_ratio_grid, n_voxels).tolist()


# In[27]:

solvers = []
for a, l in zip(alpha, l1_ratio):
    solvers.append(lm.ElasticNet(l1_ratio=l, alpha=a, positive=True, warm_start=True, max_iter=25000, fit_intercept=True, normalize=True))


# In[10]:

p = parallelization()


# In[11]:

models = p.start(Model, n, gtab_seen*n_variations, sphere=[sphere], isotropic=[BiExponentialIsotropicModel], solver=solvers)


# In[12]:

fits = p.start([i.fit for i in models], n, data_seen*n_variations, te_seen*n_variations,  g_seen*n_variations)


# In[ ]:

predicts = p.start([i.predict for i in fits], n, gtab_unseen*n_variations, te_unseen*n_variations)


# In[ ]:

#predicts_seen = p.start([i.predict for i in fits], n, gtab_seen*n_variations, te_seen*n_variations)


# In[ ]:

#filename = 'predictions_seen.csv'
#with open(filename, 'w') as f:
#   writer = csv.writer(f, delimiter=',')
#   writer.writerows(predicts_seen)


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


# In[ ]:

LSEs = np.zeros(n)
for j in range(0,n):
    LSEs[j] = utils.LSE(predicts[j], (data_unseen*n_variations)[j])


# In[ ]:

LSE_average = np.zeros(n_variations)
for m in range(n_variations):
    LSE_average[m] = np.mean(LSEs[range(m*n_voxels, (m+1)*n_voxels)])


# In[ ]:

print("Median LSE = %s"%np.mean(LSEs))

