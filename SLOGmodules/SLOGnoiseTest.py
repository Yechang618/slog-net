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

#### Import GNN packages
from SLOGmodules import graphTools as graphTools
from SLOGmodules import dataTools as dataTools
from alegnn.utils import graphML as gml

from alegnn.modules import architectures as archit
from alegnn.modules import model as model
from alegnn.modules import training as training
from alegnn.modules import evaluation as evaluation
from alegnn.modules import loss as loss
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed

### Import trained models
from SLOGTrainedModels import trainedModels as trainedModels


def to_numpy(x):
    dataType = type(x) # get data type so that we don't have to convert
    if 'numpy' in repr(dataType):
        return x
    elif 'torch' in repr(dataType):
        x1 = x.clone().detach().requires_grad_(False)
        return x1.numpy()
    
def to_torch(x):
    dataType = type(x) # get data type so that we don't have to convert
    if 'numpy' in repr(dataType):
        return torch.tensor(x)
    elif 'torch' in repr(dataType):
        return x  
    
def noiseTest_dropbox(nNodes,P,S, modelDir, **kwargs):
    ## Assertation
    
    ## Parameters loading (kwargs)
    if 'location' in kwargs.keys():
        location = kwargs['location']
    else:
        location = 'office'

    if 'modelParas' in kwargs.keys():
        modelParas = kwargs['modelParas']
    else:
        modelParas = {}

    if 'q' in modelParas.keys():
        q = modelParas['q']
    else:
        q = 4
        
    if 'simuParas' in kwargs.keys():
        simuParas = kwargs['simuParas']
    else:
        simuParas = {}        
    
    if 'alpha' in simuParas.keys():
        alpha = simuParas['alpha']
    else:
        alpha = 1.0
        simuParas['alpha'] = alpha
        
    if 'normalize_g_hat' in kwargs.keys():
        normalize_g_hat = kwargs['normalize_g_hat']
    else:
        normalize_g_hat = False       
        
    if 'N_C' in simuParas.keys(): 
        # Number of sources per signal in classification
        N_C = simuParas['N_C']
    else:
        N_C = 3
        simuParas['N_C'] = N_C    
        
    if 'nClasses' in simuParas.keys(): 
        # Number of classes (i.e. number of communities) in classification
        nClasses = simuParas['nClasses']
    else:
        nClasses = 3
        simuParas['nClasses'] = nClasses    
   
    if 'graphType' in simuParas.keys(): 
        # Number of classes (i.e. number of communities) in classification
        graphType = simuParas['graphType']
    else:
        graphType = 'ER'
        simuParas['graphType'] = graphType 

    if 'graphOptions' in simuParas.keys():
        graphOptions = simuParas['graphOptions']
    else:
        graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
        graphOptions['probIntra'] = 0.3 # Probability of drawing edges
        simuParas['graphOptions'] = graphOptions

    if 'L' in simuParas.keys():
        L = simuParas['L']
    else:
        L = 5
        simuParas['L'] = L
        
    if 'filterType' in simuParas.keys():
        filterType = simuParas['filterType']
    else:
        filterType = 'h'
        simuParas['filterType'] = filterType    
        
    if 'noiseLevel' in simuParas.keys():
        noiseLevel = simuParas['noiseLevel']
    else:
        noiseLevel = 0
        simuParas['noiseLevel'] = noiseLevel        

    if 'noiseType' in simuParas.keys():
        noiseType = simuParas['noiseType']
    else:
        noiseType = 'gaussion'
        simuParas['noiseType'] = noiseType        

    if 'C' in simuParas.keys():
        C = simuParas['C']
    else:
        C = nNodes
        simuParas['C'] = C        

    if 'K' in simuParas.keys():
        K = simuParas['K']
    else:
        K = 5
        simuParas['K'] = K        

    if 'N_realiz' in simuParas.keys():
        N_realiz = simuParas['N_realiz']
    else:
        N_realiz = 10
        simuParas['N_realiz'] = N_realiz  

    ## Model settings
    #
    if 'modelSettings' in kwargs.keys():
        modelSettings = kwargs['modelSettings']
    else:
        modelSettings = {}
        
    if 'model_number' in simuParas.keys():
        model_number = simuParas['model_number']
    else:
        model_number = 0        
        
    if 'thisLoss' in modelSettings.keys():
        thisLoss = modelSettings['thisLoss']
    else:
        thisLoss = SLOGtools.myLoss
        
    if 'thisEvaluator' in modelSettings.keys():
        thisEvaluator = modelSettings['thisEvaluator']
    else:
        thisEvaluator = SLOGevaluator.evaluate   
        
    if 'thisObject' in modelSettings.keys():
        thisObject = modelSettings['thisObject']
    else:
        thisObject = SLOGobj.myFunction_slog_1
 
    model_name = 'SLOG-Net'
    device = 'gpu'
    optimAlg = 'ADAM'
    learningRate = 0.001
    beta1 = 0.9
    beta2 = 0.999

    ## Save dir
    
    if location == 'home':
        print('Running test at ', location)
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
    elif location == 'office':
        print('Running test at ', location)        
        saveDir_dropbox = '/Users/changye/Dropbox/onlineResults/experiments'
    else:
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
        
    ## Generate modelSaveDir
    # modelDirList
    label = 'Best'
    saveDir = os.path.join(saveDir_dropbox,modelDir)
    gsoName = 'gso-' + graphType + '.npy'
    gsoDir = os.path.join(saveDir, gsoName)
    GA = np.load(gsoDir)
    d,An, eigenvalues, V = SLOGtools.get_eig_normalized_adj(GA)
    gso = An
    if model_number == 1:
        SLOG_net = SLOGarchi.GraphSLoG_v3(V,nNodes,q,K, thisObject)
    else:
        SLOG_net = SLOGarchi.GraphSLoG_v1(V,nNodes,C,K, thisObject)
    thisOptim = optim.Adam(SLOG_net.parameters(), lr = learningRate, betas = (beta1,beta2))
    thisTrainer = SLOGtrainer.slog_Trainer    
    loadedModel = SLOGmodel.Model(SLOG_net,thisLoss,thisOptim, thisTrainer,thisEvaluator, device, model_name,  None)
    loadedModel.load_from_dropBox(saveDir, label = label)        
    
    # Test begins
    result = {}
    re_x = np.zeros(N_realiz)
    re_g = np.zeros(N_realiz)    
    for n_realiz in range(N_realiz):
        X = SLOGtools.X_generate(nNodes,P,S)
        if filterType == 'g':
            g0 = SLOGtools.g_generate_gso(nNodes,alpha, eigenvalues,L)
        else:
            g0 = SLOGtools.h_generate_gso(nNodes,alpha, eigenvalues,L)
        X = to_numpy(X)
        g0 = to_numpy(g0)
        V = to_numpy(V)
        if normalize_g_hat:
            g0 = nNodes*g0/np.sum(g0)
        else:
            g0 = C*g0/np.sum(g0)
        h0 = 1./g0
        H = np.dot(V,np.dot(np.diag(h0),V.T))
        if noiseType == 'gaussion':
            noise = np.random.normal(0,1,[nNodes, P])
            noise = noise/LA.norm(noise,'fro')*LA.norm(X,'fro')
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1,1,[nNodes, P])
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(X))
        else:
            noise = np.zeros([nNodes, P])
        Y = np.dot(H,X) + noiseLevel*noise
        Y_test = to_torch(Y)
        x_hat, g_hat = loadedModel.archit(Y_test)
        g_hat = to_numpy(g_hat)  
        
        if normalize_g_hat:
            g_hat = nNodes*g_hat/np.sum(g_hat)      
        
        Z = linalg.khatri_rao(np.dot(Y.T,V),V)
#         print('Sum of g', np.sum(g_hat), np.sum(g0))
        x_recv = np.dot(Z,g_hat)
        X_recv = x_recv.reshape((P,nNodes)).T
        re_x_1 = LA.norm(X_recv - X,'fro')/LA.norm(X,'fro')
        re_g_1 = LA.norm(g0 - g_hat)/LA.norm(g0)
        re_x_2 = LA.norm(X_recv + X,'fro')/LA.norm(X,'fro')
        re_g_2 = LA.norm(g0 + g_hat)/LA.norm(g0) 
        if re_g_1 > re_g_2:
            re_g[n_realiz] = re_g_2
            re_x[n_realiz] = re_x_2
        else:
            re_g[n_realiz] = re_g_1
            re_x[n_realiz] = re_x_1               
    result['re_x'] = re_x    
    result['re_g'] = re_g     
    
    return result

def noiseClassificationTest_dropbox(nNodes,P,S, modelDir, **kwargs):
    ## Assertation
    
    ## Parameters loading (kwargs)
    if 'location' in kwargs.keys():
        location = kwargs['location']
    else:
        location = 'office'

    if 'modelParas' in kwargs.keys():
        modelParas = kwargs['modelParas']
    else:
        modelParas = {}

    if 'q' in modelParas.keys():
        q = modelParas['q']
    else:
        q = 4
        
    if 'simuParas' in kwargs.keys():
        simuParas = kwargs['simuParas']
    else:
        simuParas = {}        
    
    if 'alpha' in simuParas.keys():
        alpha = simuParas['alpha']
    else:
        alpha = 1.0
        simuParas['alpha'] = alpha
        
    if 'normalize_g_hat' in kwargs.keys():
        normalize_g_hat = kwargs['normalize_g_hat']
    else:
        normalize_g_hat = False       
        
    if 'N_C' in simuParas.keys(): 
        # Number of sources per signal in classification
        N_C = simuParas['N_C']
    else:
        N_C = 3
        simuParas['N_C'] = N_C    
        
    if 'nClasses' in simuParas.keys(): 
        # Number of classes (i.e. number of communities) in classification
        nClasses = simuParas['nClasses']
    else:
        nClasses = 4
        simuParas['nClasses'] = nClasses    
   
    if 'graphType' in simuParas.keys(): 
        # Number of classes (i.e. number of communities) in classification
        graphType = simuParas['graphType']
    else:
        graphType = 'ER'
        simuParas['graphType'] = graphType 

    if 'graphOptions' in simuParas.keys():
        graphOptions = simuParas['graphOptions']
    else:
        graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
        graphOptions['probIntra'] = 0.3 # Probability of drawing edges
        simuParas['graphOptions'] = graphOptions

    if 'L' in simuParas.keys():
        L = simuParas['L']
    else:
        L = 5
        simuParas['L'] = L
        
    if 'device' in simuParas.keys():
        device = simuParas['device']
    else:
        device = 'cpu'
      
        
    if 'filterType' in simuParas.keys():
        filterType = simuParas['filterType']
    else:
        filterType = 'h'
        simuParas['filterType'] = filterType    
        
    if 'noiseLevel' in simuParas.keys():
        noiseLevel = simuParas['noiseLevel']
    else:
        noiseLevel = 0
        simuParas['noiseLevel'] = noiseLevel        

    if 'noiseType' in simuParas.keys():
        noiseType = simuParas['noiseType']
    else:
        noiseType = 'gaussion'
        simuParas['noiseType'] = noiseType        

    if 'C' in simuParas.keys():
        C = simuParas['C']
    else:
        C = nNodes
        simuParas['C'] = C        

    if 'K' in simuParas.keys():
        K = simuParas['K']
    else:
        K = 5
        simuParas['K'] = K        

    if 'N_realiz' in simuParas.keys():
        N_realiz = simuParas['N_realiz']
    else:
        N_realiz = 10
        simuParas['N_realiz'] = N_realiz  

    ## Model settings
    #
    if 'modelSettings' in kwargs.keys():
        modelSettings = kwargs['modelSettings']
    else:
        modelSettings = {}
        
    if 'model_number' in simuParas.keys():
        model_number = simuParas['model_number']
    else:
        model_number = 0        
        
    if 'thisLoss' in modelSettings.keys():
        thisLoss = modelSettings['thisLoss']
    else:
        thisLoss = SLOGtools.myLoss
        
    if 'thisEvaluator' in modelSettings.keys():
        thisEvaluator = modelSettings['thisEvaluator']
    else:
        thisEvaluator = SLOGevaluator.evaluate   
        
    if 'thisObject' in modelSettings.keys():
        thisObject = modelSettings['thisObject']
    else:
        thisObject = SLOGobj.myFunction_slog_1
    ## Classification settings
    nTest = P
    if 'signalMode' in simuParas.keys():
        signalMode = simuParas['signalMode']
    else:
        signalMode = 'Gaussion'
        
    if 'filterMode' in simuParas.keys():
        filterMode = simuParas['filterMode']
    else:
        filterMode = 'default'
        
    if 'selectMode' in simuParas.keys():
        selectMode = simuParas['selectMode']
    else:
        selectMode = 'random'

    if 'tMax' in simuParas.keys():
        tMax = simuParas['tMax']
    else:
        tMax = N
         
    model_name = 'SLOG-Net'
    device = 'gpu'
    optimAlg = 'ADAM'
    learningRate = 0.001
    beta1 = 0.9
    beta2 = 0.999

    ## Save dir
    
    if location == 'home':
        print('Running test at ', location)
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
    elif location == 'office':
        print('Running test at ', location)        
        saveDir_dropbox = '/Users/changye/Dropbox/onlineResults/experiments'
    else:
        saveDir_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineResults\experiments"
        
    ## Generate modelSaveDir
    # modelDirList
    label = 'Best'
    saveDir = os.path.join(saveDir_dropbox,modelDir)
    gsoName = 'gso-' + graphType + '.npy'
    cLabelsName = 'cLabels-' + graphType + '.npy'
    gsoDir = os.path.join(saveDir, gsoName)
    cLabelsDir = os.path.join(saveDir, cLabelsName)
    GA = np.load(gsoDir)
    d,An, eigenvalues, V = SLOGtools.get_eig_normalized_adj(GA)    
    gso = An
    communityLabels = np.load(cLabelsDir)
    temp_result = SLOGtools.community_LabelsToNodeSets(communityLabels,GA,S)
    sourceNodes = temp_result['sourceNodes']
    communityNodeList = temp_result['communityList'] # Node list for each of the communities
    
    nClass = temp_result['nClass']    
    
    data = SLOGdata.noiseClassificationTest_generator(gso, nTest, S, communityLabels, V, eigenvalues, L = L, tMax = tMax, alpha = alpha, filterMode = filterMode, selectMode = selectMode, signalMode = signalMode, filterType = filterType, noiseLevel = noiseLevel)    
    
    # Load SLOG-Net
    if model_number == 1:
        SLOG_net = SLOGarchi.GraphSLoG_v3(V,nNodes,q,K, thisObject)
    else:
        SLOG_net = SLOGarchi.GraphSLoG_v1(V,nNodes,C,K, thisObject)
    thisOptim = optim.Adam(SLOG_net.parameters(), lr = learningRate, betas = (beta1,beta2))
    thisTrainer = SLOGtrainer.slog_Trainer    
    loadedModel = SLOGmodel.Model(SLOG_net,thisLoss,thisOptim, thisTrainer,thisEvaluator, device, model_name,  saveDir)
    loadedModel.load_from_dropBox(saveDir, label = label)        
    
    # Load CrsGNN
    
    # Model Parameters
    hParamsSelGNN = {} # Create the dictionary to save the hyperparameters
    hParamsSelGNN['name'] = 'SelGNN' # Name the architecture
    hParamsSelGNN['F'] = [1, 5, 5] # Features per layer (first element is the number of input features)
    hParamsSelGNN['K'] = [3, 3] # Number of filter taps per layer
    hParamsSelGNN['bias'] = True # Decide whether to include a bias term
    hParamsSelGNN['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsSelGNN['rho'] = gml.MaxPoolLocal # Summarizing function
    hParamsSelGNN['alpha'] = [2, 3] # alpha-hop neighborhood that
    hParamsSelGNN['N'] = [10, 5] # Number of nodes to keep at the end of each layer is affected by the summary
    hParamsSelGNN['order'] = 'Degree'
    hParamsSelGNN['dimLayersMLP'] = [nClasses] # Dimension of the fully connected layers after the GCN layers
    hParamsCrsGNN = deepcopy(hParamsSelGNN)
    hParamsCrsGNN['name'] = 'CrsGNN'
    hParamsCrsGNN['rho'] = nn.MaxPool1d
    hParamsCrsGNN['order'] = None # We don't need any special ordering, since
    thisName = hParamsCrsGNN['name']

        #\\\ Architecture
    # Load seed
    SLOGtools.loadSeed(saveDir)
    coarsened_outputs_save_name = 'CrsGNNcoarsen-' + graphType + '.npy'
    coarsened_outputs_save_dir_dropBox = os.path.join(saveDir, coarsened_outputs_save_name) 
    coarsend_outputs = torch.load(coarsened_outputs_save_dir_dropBox)
    thisArchit = archit.SelectionGNN(# Graph filtering
                                 hParamsCrsGNN['F'],
                                 hParamsCrsGNN['K'],
                                 hParamsCrsGNN['bias'],
                                 # Nonlinearity
                                 hParamsCrsGNN['sigma'],
                                 # Pooling
                                 hParamsCrsGNN['N'],
                                 hParamsCrsGNN['rho'],
                                 hParamsCrsGNN['alpha'],
                                 # MLP
                                 hParamsCrsGNN['dimLayersMLP'],
                                 # Structure
                                 gso,
                                 coarsening = True,
                                load_coarsened_GSO = True,
                                 # Coarsened outputs
                                coarsened_outputs = coarsend_outputs          
                                    )
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
    print(device)
#     thisArchit.to(device)

    #\\\ Optimizer
    thisOptim = optim.Adam(thisArchit.parameters(), lr = learningRate, betas = (beta1,beta2))
    trainer = training.Trainer
    evaluator = evaluation.evaluate
    lossFunction = nn.CrossEntropyLoss        
        #\\\ Model
    CrsGNN = model.Model(thisArchit,
                     lossFunction(),
                     thisOptim,
                     trainer,
                     evaluator,
                     device,
                     thisName,
                     saveDir,
                     saveDir_dropbox = saveDir)
    CrsGNN.load_from_dropBox(saveDir, label = label)  
        
        
        
        
        
    # Test begins
    result = {}
    acc_slogbest_top1 = np.zeros(N_realiz)
    acc_slogbest_topN = np.zeros(N_realiz)
    acc_sloglast_top1 = np.zeros(N_realiz)
    acc_sloglast_topN = np.zeros(N_realiz)    
    acc_crs_best = np.zeros(N_realiz)
    acc_crs_last = np.zeros(N_realiz)    
    for n_realiz in range(N_realiz):
        data.renew_testSet()
        CrsEvalVars = CrsGNN.evaluate(data)
        costBest_crs = CrsEvalVars['costBest']
        costLast_crs = CrsEvalVars['costLast']
        result_evaluate_slog_topN = loadedModel.evaluate(data, topN = S)
        result_evaluate_slog_top1 = loadedModel.evaluate(data, topN = 1)        
        costBest_slog_topN = result_evaluate_slog_topN['costBest']
        costLast_slog_topN = result_evaluate_slog_topN['costBest']    
        costBest_slog_top1 = result_evaluate_slog_topN['costBest']
        costLast_slog_top1 = result_evaluate_slog_topN['costBest']
        acc_slogbest_top1[n_realiz] = result_evaluate_slog_top1['costBest']
        acc_slogbest_topN[n_realiz] = result_evaluate_slog_topN['costBest']
        acc_sloglast_top1[n_realiz] = result_evaluate_slog_top1['costLast']  
        acc_sloglast_topN[n_realiz] = result_evaluate_slog_topN['costLast']  
        acc_crs_best[n_realiz] = costBest_crs
        acc_crs_last[n_realiz] = costLast_crs          
        print('CrsGNN:', costBest_crs)     
        print('CrsGNN:', costLast_crs)
        print('SLOG topN:',result_evaluate_slog_topN) #{'costBest': 0.0, 'costLast': 0.0}
        print('SLOG top1:',result_evaluate_slog_top1)
    result['acc_slogbest_top1'] = acc_slogbest_top1 
    result['acc_slogbest_topN'] = acc_slogbest_topN    
    result['acc_sloglast_top1'] = acc_sloglast_top1
    result['acc_sloglast_topN'] = acc_sloglast_topN  
    result['acc_crs_best'] = acc_crs_best
    result['acc_crs_last'] = acc_crs_last      
    return result
    
 