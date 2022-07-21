import numpy as np
from numpy import linalg as LA
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
# import mymodule as myModules
import matplotlib.cm as cm
from scipy import linalg
from timeit import default_timer as timer
import networkx as nx
import os
import pickle
import datetime
#### Import SLOG packakes
from SLOGmodules import SLOGtools as SLOGtools
from SLOGmodules import SLOGobjective as SLOGobj
from SLOGmodules import SLOGarchitectures as SLOGarchi 
from SLOGmodules import SLOGtraining as SLOGtrainer
from SLOGmodules import SLOGmodel as SLOGmodel
from SLOGmodules import SLOGevaluation as SLOGevaluator
from SLOGmodules import SLOGdata as SLOGdata
from SLOGmodules import SLOGSaveDir as SLOGSaveDir
from SLOGmodules import SLOGnoiseTest as SLOGnoiseTest

#### Import GNN packages
from SLOGmodules import graphTools as graphTools
from SLOGmodules import dataTools as dataTools


from alegnn.modules import architectures as archit
from alegnn.modules import model as model
from alegnn.modules import training as training
from alegnn.modules import evaluation as evaluation
from alegnn.modules import loss as loss
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed

### Import trained models
from SLOGTrainedModels import trainedModels as trainedModels

## Load saveSettings
location = 'home'
# location = 'office'
saveDir_result = SLOGSaveDir(location)
saveDir_dropbox = saveDir_result['saveDir_dropbox']
### ER
nNodes = 20 # Number of nodes
P = 400
S = 2

nClasses = 4 # Number of classes (i.e. number of communities)
N_C = S # Number of sources per signal
alpha = 1.0

graphType = 'SBM'
list_number = 1
model_list = trainedModels.get_modelList(graphType, list_number)

# list_name = 'modelDirList_0' # Model v3,Train filter type = h (list_number = 3) q = 1, noise = 0.0
# list_name = 'modelDirList_1' # Model v3,Train filter type = h (list_number = 3) q = 1, noise = 0.0
# list_name = 'modelDirList_test' # Model v3,Train filter type = h (list_number = 3) q = 1, noise = 0.0
# list_name = 'modelDirList_2' # Model v3,Train filter type = h (list_number = 3) q = 1, noise = 0.0, 200k
# list_name = 'modelDirList_3' # Model v3,Train filter type = h (list_number = 3) q = 1, noise = 0.0, 200k at home
# list_name = 'modelDirList_4' # Model v3,Train filter type = h (list_number = 3) q = 1, noise = 0.05 (100k at home)
# list_name = 'modelDirList_5' # Model v3,Train filter type = wt (list_number = 3) q = 1, noise = 0.05 (100k at office)
list_name = 'modelDirList_6' # Model v3,Train filter type = wt (list_number = 3) q = 1, noise = 0.05 (200k at office)

print(model_list)

## Model number: 0, SLOG-Net_v1; 1, SLOG-Net_v3
model_number = 1
# model_number = 0
q = 4
# q = 10
# q = 1

modelDirList = model_list[list_name]
# modelDirList = model_list['modelDirList_1']
# modelDirList = model_list['modelDirList_2']

print(model_list[list_name+'_label'])
## 
normalize_g_hat = True # Normalize g_hat to sum = nNodes. suggest nnly mormalize g_hat for 'ER' list 1
# normalize_g_hat = False 

## GPU
useGPU = True
if useGPU and torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
# Notify:
device = torch.device(device)
print("Device selected: %s" % device)

### Model parameters
K = 5 # number of layers
# C = 1 # constrain constant
C = nNodes # constrain constant
# filterTrainType = 'g'
# filterTrainType = 'h'
filterTrainType = 'wt'
### Simulation parameters
simuParas = {}

nTrain_slog = 100000
batchsize_slog = 400
nValid = batchsize_slog
nTest = batchsize_slog
nEpochs = 50
N_realiz = 500
### Data parameters
L = 5
alpha = 1.0
N_noiseLvs = 10
d_noiseLvs = 0.1
n0 = 0



### Graph parameters


simuParas['nNodes'] = nNodes
simuParas['nClasses'] = nClasses
simuParas['N_C'] = N_C

graphType = 'SBM' # Type of graph
graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
graphOptions['nCommunities'] = nClasses # Number of communities
graphOptions['probIntra'] = 0.8 # Probability of drawing edges intra communities
graphOptions['probInter'] = 0.2 # Probability of drawing edges inter communities


# graphType = 'BA' # Type of graph
# graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
# graphOptions['alpha'] = 1.0

# graphType = 'ER' # Type of graph
# graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
# graphOptions['probIntra'] = 0.3 # Probability of drawing edges

simuParas['graphType'] = graphType
simuParas['graphOptions'] = graphOptions

## Filter type: g or h
# filterType = 'g'
# filterType = 'h'
filterType = 'wt'

## Signal mode: gaussion or 1
signalMode = 'Gaussion'

## Train mode: Wt or default (Vdiag(\tilde{h})V^T)
trainMode = 'default'
# trainMode = 'Wt'

## Filter mode: Wt or default (Vdiag(\tilde{h})V^T)
filterMode = 'default'
# filterMode = 'Wt'

## Selection mode: random or nodes with top-N_C degree
selectMode = 'random'

## Noise level
noiseLevel = 0.0
noiseType = 'uniform'


simuParas['device'] = device
simuParas['nTrain_slog'] = nTrain_slog
simuParas['batchsize_slog'] = batchsize_slog
simuParas['nValid'] = nValid
simuParas['nTest'] = nTest
simuParas['L'] = L
simuParas['noiseLevel'] = noiseLevel
simuParas['noiseType'] = noiseType
simuParas['filterType'] = filterType
simuParas['signalMode'] = signalMode
simuParas['trainMode'] = trainMode
simuParas['filterMode'] = filterMode
simuParas['selectMode'] = selectMode
simuParas['graphType'] = graphType 
simuParas['alpha'] = alpha
simuParas['nEpochs'] = nEpochs
simuParas['C'] =  C
simuParas['K'] =  K
simuParas['N_realiz'] =  N_realiz
simuParas['model_number'] = model_number
modelParas = {}
modelParas['q'] = q

tMax = None # Maximum number of diffusion times (W^t for t < tMax)
tMax = 5
simuParas['tMax'] = tMax

## Experiment results
# modelDirList = modelDirList_0
# modelDirList = modelDirList_1

N_model = len(modelDirList)
noiseLvs = n0 + d_noiseLvs*np.arange(N_noiseLvs)
result_acc_slogBest_topN = np.zeros([N_noiseLvs, N_model,N_realiz])
result_acc_slogBest_top1 = np.zeros([N_noiseLvs, N_model,N_realiz])
result_acc_crsgnnBest = np.zeros([N_noiseLvs, N_model,N_realiz])
n_model = 0
for modelDir in modelDirList:
    for n_nlvs in range(N_noiseLvs):
        simuParas['noiseLevel'] = noiseLvs[n_nlvs]        
        print('Model ', n_model, ', in ', modelDir,',noiseLevel = ', noiseLvs[n_nlvs])
        result = SLOGnoiseTest.noiseClassificationTest_dropbox(nNodes,P,S, modelDir, simuParas = simuParas,modelParas = modelParas, location = location , normalize_g_hat = normalize_g_hat)
        result_acc_slogBest_top1[n_nlvs,n_model,:] = result['acc_slogbest_top1']
        result_acc_slogBest_topN[n_nlvs,n_model,:]  = result['acc_slogbest_topN']
#         result['acc_sloglast_top1'] = acc_sloglast_top1
#         result['acc_sloglast_topN'] = acc_sloglast_topN  
        result_acc_crsgnnBest[n_nlvs,n_model,:] = result['acc_crs_best']
#         result['acc_crs_last'] = acc_crs_last  
    n_model += 1
    
    
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
model = 0

# model_index = [0,1,2,3,4,5,6,7,8,9]
model_index = [0,6]
# model_index = [0,1,2,3,4]
# model_index = [3]
# N_noiseLvIndex = len(noiseLv_index)
N_model_1 = len(model_index)

result_acc_slogBest_top1_1 = result_acc_slogBest_top1[:,model_index,:]
result_acc_slogBest_topN_1 = result_acc_slogBest_topN[:,model_index,:]
result_acc_crsgnnBest_1 = result_acc_crsgnnBest[:,model_index,:]

N_noiseLvIndex = N_noiseLvs

print(result_acc_slogBest_top1_1.shape)
# re_x_avg = np.mean(np.mean(result_exp_rex_1, axis=2), axis=1)
# re_g_avg = np.mean(np.mean(result_exp_reg_1, axis=2), axis=1)
result_acc_slogBest_top1_avg = np.mean(result_acc_slogBest_top1_1.reshape((N_noiseLvs,N_model_1*N_realiz)), axis=1)
result_acc_slogBest_topN_avg = np.mean(result_acc_slogBest_topN_1.reshape((N_noiseLvs,N_model_1*N_realiz)), axis=1)
result_acc_crsgnnBest_avg = np.mean(result_acc_crsgnnBest_1.reshape((N_noiseLvs,N_model_1*N_realiz)), axis=1)
result_acc_slogBest_top1_std = np.std(result_acc_slogBest_top1_1.reshape((N_noiseLvs,N_model_1*N_realiz)), axis=1)
result_acc_slogBest_topN_std = np.std(result_acc_slogBest_topN_1.reshape((N_noiseLvs,N_model_1*N_realiz)), axis=1)
result_acc_crsgnnBest_std = np.std(result_acc_crsgnnBest_1.reshape((N_noiseLvs,N_model_1*N_realiz)), axis=1)

result_acc_slogBest_top1_reshape = result_acc_slogBest_top1_1.reshape((N_noiseLvs,N_model_1*N_realiz))
result_acc_slogBest_topN_reshape = result_acc_slogBest_topN_1.reshape((N_noiseLvs,N_model_1*N_realiz))
result_acc_crsgnnBest_reshape = result_acc_crsgnnBest_1.reshape((N_noiseLvs,N_model_1*N_realiz))

print(np.mean(result_acc_slogBest_top1_1, axis=2).shape)

print_date = '0721'
print('Print date: ', print_date)

if model_number == 0:
    comment = '(SLOG-Net v1, N_realiz = '+str(N_realiz)+ ')'
elif model_number == 1:
    comment = '(SLOG-Net v3, N_realiz = '+str(N_realiz)+ ')'
else:
    comment = '(Unknown)'
comment_additional = '(q=' + str(q) + ')(train w. wt)(noise = 0.00)'
# comment_additional = '(train w. g)'
comment = comment + comment_additional + '('+ str(model_index) + ')'
plot_num = "0"

fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True,figsize=(10,10))
N_noiseLvIndex = 10
axes[0].errorbar(noiseLvs[:N_noiseLvIndex],1 - result_acc_slogBest_topN_avg[:N_noiseLvIndex], yerr =result_acc_slogBest_topN_std[:N_noiseLvIndex], fmt = 'ro')
axes[0].set_title('Mean accuracy of top-N '+ comment)
axes[0].set_xlabel('Noise level')
axes[0].set_ylabel('Mean accuracy')
axes[0].set_ylim([0,1.0])
axes[0].grid()
axes[1].errorbar(noiseLvs[:N_noiseLvIndex],1 - result_acc_slogBest_top1_avg[:N_noiseLvIndex], yerr =result_acc_slogBest_top1_std[:N_noiseLvIndex], fmt = 'ro')
axes[1].set_title('Mean accuracy of top-1 '+ comment)
axes[1].set_xlabel('Noise level')
axes[1].set_ylabel('Mean accuracy')
axes[1].set_ylim([0,1.0])
axes[1].grid()
axes[2].errorbar(noiseLvs[:N_noiseLvIndex],1 - result_acc_crsgnnBest_avg[:N_noiseLvIndex], yerr =result_acc_crsgnnBest_std[:N_noiseLvIndex], fmt = 'ro')
axes[2].set_title('Mean accuracy of CrsGNN'+ '('+ str(model_index) + ')')
axes[2].set_xlabel('Noise level')
axes[2].set_ylabel('Mean accuracy')
axes[2].set_ylim([0,1.0])
axes[2].grid()
plt.show()

saveDir_plots = os.path.join(saveDir_dropbox_root,'plots')
save_name = "fig_slognoiseclassification_n" + str(N_realiz) + '_' + print_date+ "_list_" + str(list_number) + "_model_" + str(model_number) + "_" + list_name + "_errorbar_v1_"+plot_num+".jpg"

saveDir_fig = os.path.join(saveDir_plots,save_name)
fig.savefig(saveDir_fig,format='jpg')

fig, axes = plt.subplots(nrows=3, ncols=1, constrained_layout=True,figsize=(10,10))
# N_noiseLvIndex = 9
y_label_list = n0 + d_noiseLvs*np.arange(N_noiseLvIndex)
pos_1 = axes[0].imshow(np.mean(result_acc_slogBest_topN_1,axis = 2), cmap=cm.seismic)
axes[0].set_title('Mean accuracy of SLOG-topN average over realizations '+ comment)
axes[0].set_ylabel('NoiseLv')
axes[0].set_yticks(np.arange(N_noiseLvIndex))
axes[0].set_yticklabels([str(round(float(label), 2)) for label in y_label_list])
axes[0].set_xlabel('Model')
pos_2 = axes[1].imshow(np.mean(result_acc_slogBest_top1_1,axis = 2), cmap=cm.seismic)
axes[1].set_title('Mean accuracy of SLOG-top1 average over realizations '+ comment)
axes[1].set_ylabel('NoiseLv')
axes[1].set_yticks(np.arange(N_noiseLvIndex))
axes[1].set_yticklabels([str(round(float(label), 2)) for label in y_label_list])
axes[1].set_xlabel('Model')
pos_3 = axes[2].imshow(np.mean(result_acc_crsgnnBest_1,axis = 2), cmap=cm.seismic)
axes[2].set_title('Mean accuracy of CrsGNN average over realizations ')
axes[2].set_ylabel('NoiseLv')
axes[2].set_yticks(np.arange(N_noiseLvIndex))
axes[2].set_yticklabels([str(round(float(label), 2)) for label in y_label_list])
axes[2].set_xlabel('Model')
fig.colorbar(pos_1, ax=axes[0], location='right', anchor=(1, 1), shrink=1.0)
fig.colorbar(pos_2, ax=axes[1], location='right', anchor=(1, 1), shrink=1.0)
fig.colorbar(pos_3, ax=axes[2], location='right', anchor=(1, 1), shrink=1.0)
plt.show()

saveDir_plots = os.path.join(saveDir_dropbox_root,'plots')
save_name = "fig_slognoiseclassification_n" + str(N_realiz) + '_' +print_date+ "_list_" + str(list_number) + "_model_" + str(model_number)+ "_" + list_name  + "_heatmap_v1_"+plot_num+".jpg"

saveDir_fig = os.path.join(saveDir_plots,save_name)
fig.savefig(saveDir_fig,format='jpg')
