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


class slog_experiments():
    def __init__(self, simuParas = None, 
                 graphOptions = None,  **kwargs):
        
        # Simulation parameters
        self.simuParas = simuParas
        self.nTrain_slog = simuParas['nTrain_slog']
        self.batchsize_slog = simuParas['batchsize_slog']
        self.nValid = simuParas['nValid']
        self.nTest = simuParas['nTest']
        self.L = simuParas['L']
        self.noiseLevel = simuParas['noiseLevel']
        self.noiseType = simuParas['noiseType']
        self.filterType = simuParas['filterType']
        self.signalMode = simuParas['signalMode']
        self.trainMode = simuParas['trainMode']
        self.filterMode = simuParas['filterMode']
        self.selectMode = simuParas['selectMode']
        self.nNodes = simuParas['nNodes']
        self.nClasses = simuParas['nClasses']
        self.N_C = simuParas['N_C']
        self.S = simuParas['S']
        self.graphType = simuParas['graphType']
        self.tMax = simuParas['tMax']
        self.alpha = simuParas['alpha']
        self.nEpochs = simuParas['nEpochs']
        if 'model_number' in simuParas.keys():
            self.model_number = simuParas['model_number']
        else:
            self.model_number = 0
                
        # Graph options
        self.graphOptions = graphOptions
        
        # Model parameters (optional)
        if 'modelParas' in kwargs.keys():
            self.modelParas = kwargs['modelParas']
            self.C = self.modelParas['C']
            self.K = self.modelParas['K']
            self.filterTrainType = self.modelParas['filterTrainType']
        else:
            self.C = self.nNodes
            self.K = 5
            self.filterTrainType = 'g'
            self.modelParas = {}
            self.modelParas['C']= self.C
            self.modelParas['K']= self.K
            self.modelParas['filterTrainType']= self.filterTrainType
        if 'q' in self.modelParas.keys():
            self.q = self.modelParas['q']
        else:
            self.q = 4
            
        # Experiment parameters (optional)
        if 'expParas' in kwargs.keys():
            self.expParas = kwargs['expParas']
            self.nRealiz = self.expParas['nRealiz']
        else:
            self.expParas = {}
            self.nRealiz = 1
            self.expParas['nRealiz'] = self.nRealiz
            
        ## Save settings (optional)
        # Including:
        # thisFilename_SLOG = 'sourceLocSLOGNET'
        # saveDirRoot = 'experiments'
        # saveDir = 'experiments/sourceLocSLOGNET'
        # saveDirRoot_dropbox = '/Users/changye/Dropbox'
        # saveDir_dropbox = '/Users/changye/Dropbox/experiments'
        if 'saveSettings' in kwargs.keys():
            self.saveSettings = kwargs['saveSettings']
            self.thisFilename_SLOG = self.saveSettings['thisFilename_SLOG']
            self.saveDirRoot = self.saveSettings['saveDirRoot'] # Relative location where to save the file
            self.saveDir = self.saveSettings['saveDir']# Dir where to save all the results from each run
            self.saveDirRoot_dropbox = self.saveSettings['saveDirRoot_dropbox']
            self.saveDir_dropbox = self.saveSettings['saveDir_dropbox']         
        else:
            self.thisFilename_SLOG = 'sourceLocSLOGNET'
            self.saveDirRoot = 'experiments' # Relative location where to save the file
            self.saveDir = os.path.join(self.saveDirRoot, self.thisFilename_SLOG) # Dir where to save all the results from each run
            saveDirRoot_dropbox = '/Users/changye'
            saveDirRoot_dropbox = os.path.join(saveDirRoot_dropbox, 'Dropbox')
            self.saveDirRoot_dropbox = os.path.join(saveDirRoot_dropbox, 'onlineResults')
            self.saveDir_dropbox = os.path.join(self.saveDirRoot_dropbox, self.saveDir)
            
            self.saveSettings = {}
            self.saveSettings['thisFilename_SLOG'] = self.thisFilename_SLOG            
            self.saveSettings['saveDirRoot'] = self.saveDirRoot
            self.saveSettings['saveDir'] = self.saveDir
            self.saveSettings['saveDirRoot_dropbox'] = self.saveDirRoot_dropbox
            self.saveSettings['saveDir_dropbox'] = self.saveDir_dropbox
        
        self.experiment_results = []
        for i in range(self.nRealiz):
            result_i = self.run_single_experiment()
            self.experiment_results.append(result_i)
       
    def get_experiment_result(self):
        return self.experiment_results
            
    def run_single_experiment(self,**kwargs):    
        ## kwargs:

        #\\\ Create .txt to store the values of the setting parameters for easier
        # reference  when running multiple experiments
        today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Append date and time of the run to the directory, to avoid several runs of
        # overwritting each other.
        saveDir = self.saveDir + '-' + self.graphType + '-' + today
        saveDir_dropbox = self.saveDir_dropbox + '-' + self.graphType + '-' + today
        # 
        saveDirs = {}
        saveDirs['saveDir'] = saveDir
        saveDirs['saveDir_dropbox'] = saveDir_dropbox

        # Create directory
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        if not os.path.exists(saveDir_dropbox):
            print("Create: ",saveDir_dropbox)
            os.makedirs(saveDir_dropbox)    

        useGPU = True
        if useGPU and torch.cuda.is_available():
            device = 'cuda:0'
            torch.cuda.empty_cache()
        else:
            device = 'cpu'
        # Notify:
        print("Device selected: %s" % device)   

        # Create the file where all the (hyper)parameters are results will be saved.
        varsFile = os.path.join(saveDir,'hyperparameters.txt')
        with open(varsFile, 'w+') as file:
            file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        # Parameter pass
#         nNodes = self.nNodes
#         graphType = self.graphType
#         graphOptions = self.graphOptions
#         simuParas = self.simuParas
#         saveDirs = self.saveDirs
#         nTest = self.nTest
#         nValid = self.nValid
#         nTrain_slog = self.nTrain_slog
#         tMax = self.tMax
#         nClasses = self.nClasses
#         useGPU = self.useGPU
        #\\\ Save values:
        writeVarValues(varsFile, {'nNodes': self.nNodes, 'graphType': self.graphType})
        writeVarValues(varsFile, self.graphOptions)
        writeVarValues(varsFile, self.simuParas)
        writeVarValues(varsFile, self.modelParas)           
        writeVarValues(varsFile, saveDirs)
        writeVarValues(varsFile, {'nTrain_slog': self.nTrain_slog,
                                  'nValid': self.nValid,
                                  'nTest': self.nTest,
                                  'tMax': self.tMax,
                                  'nClasses': self.nClasses,
                                  'useGPU': useGPU})


        # Create the file where all the (hyper)parameters are results will be saved.
        varsFile = os.path.join(saveDir_dropbox,'hyperparameters.txt')
        with open(varsFile, 'w+') as file:
            file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

        #\\\ Save values:
        writeVarValues(varsFile, {'nNodes': self.nNodes, 'graphType': self.graphType})
        writeVarValues(varsFile, self.graphOptions)
        writeVarValues(varsFile, self.simuParas)
        writeVarValues(varsFile, self.modelParas)        
        writeVarValues(varsFile, saveDirs)
        writeVarValues(varsFile, {'nTrain': self.nTest,
                                  'nValid': self.nValid,
                                  'nTest': self.nTest,
                                  'tMax': self.tMax,
                                  'nClasses': self.nClasses,
                                  'useGPU': useGPU})
        
        if 'optimAlg' in kwargs.keys():
            optimAlg = kwargs['optimAlg']
        else:
            optimAlg = 'ADAM'
            
        if 'learningRate' in kwargs.keys():
            learningRate = kwargs['learningRate']
        else:
            learningRate = 0.01
            
        if 'beta1' in kwargs.keys():
            beta1 = kwargs['beta1']
        else:
            beta1 = 0.9
            
        if 'beta2' in kwargs.keys():
            beta2 = kwargs['beta2']
        else:
            beta2 = 0.999
            
        ## Graph generation
        G = SLOGtools.Graph(self.graphType, self.nNodes, self.graphOptions, save_dir = saveDir, save_dir_dropBox = saveDir_dropbox)
        G.computeGFT()
        sourceNodes, communityLabels,communityList = SLOGtools.computeSourceNodes_slog(G.A, self.nClasses, self.N_C, mode = 'random')
        d_slog,An_slog, eigenvalues_slog, V_slog   = SLOGtools.get_eig_normalized_adj(G.A)
        
        ## Data generation
#         data = SLOGdata.SLOG_ClassificationData(G, self.nTrain_slog, self.nValid, self.nTest, sourceNodes, communityList, communityLabels, V_slog, eigenvalues_slog, L = self.L, tMax = self.tMax, alpha = self.alpha, filterMode = self.filterMode, selectMode = self.selectMode, signalMode = self.signalMode, filterType = self.filterType, noiseLevel = self.noiseLevel)
        data = SLOGdata.SLOG_GeneralData(G, self.nTrain_slog, self.nValid, self.nTest, self.S, V_slog, eigenvalues_slog, L = self.L, alpha = self.alpha,filterType = self.filterType, noiseLevel = self.noiseLevel, noiseType = self.noiseType)
#         data.astype(torch.float64)
        data.expandDims()
        
        C = self.C
        K = self.K
        filterTrainType = self.filterTrainType #'g'
        thisLoss = SLOGtools.myLoss
        thisEvaluator = SLOGevaluator.evaluate
        
        if self.model_number == 1:
            thisObject = SLOGobj.myFunction_slog_3
            SLOG_net = SLOGarchi.GraphSLoG_v3(V_slog,self.nNodes,self.q,self.K, thisObject)        
        else:
            thisObject = SLOGobj.myFunction_slog_1
            SLOG_net = SLOGarchi.GraphSLoG_v1(V_slog,self.nNodes,self.C,self.K, thisObject)

        # thisObject = SLOGobj.myFunction_slog_2
        # SLOG_net = SLOGarchi.GraphSLoG_v2(V_slog,nNodes,C,K, thisObject)


        model_name = 'SLOG-Net'
# optimAlg = 'ADAM'
# learningRate = 0.001
# beta1 = 0.9
# beta2 = 0.999

        thisOptim = optim.Adam(SLOG_net.parameters(), lr = learningRate, betas = (beta1,beta2))
        thisTrainer = SLOGtrainer.slog_Trainer

        myModel = SLOGmodel.Model(SLOG_net,thisLoss,thisOptim, thisTrainer,thisEvaluator,device, model_name,  saveDir, saveDir_dropbox = saveDir_dropbox)

        result_train = myModel.train(data,self.nEpochs, self.batchsize_slog, validationInterval = 40,trainMode = self.trainMode,tMax = self.tMax, filterTrainType = self.filterTrainType) # model, data, nEpochs, batchSize
        
        best_model = result_train['bestModel']
        minLossValid = result_train['minLossValid']
        minLossTrain = result_train['minLossTrain']
        
        writeVarValues(varsFile, result_train)
            
        results = {}
        results['model'] = myModel
        results['training result'] = result_train
        
        return results

    
class slog_classification_experiments():
    def __init__(self, simuParas = None, 
                 graphOptions = None,  **kwargs):
        
        # Simulation parameters
        self.simuParas = simuParas
        self.nTrain_slog = simuParas['nTrain_slog']
        self.batchsize_slog = simuParas['batchsize_slog']
        self.nValid = simuParas['nValid']
        self.nTest = simuParas['nTest']
        self.L = simuParas['L']
        self.noiseLevel = simuParas['noiseLevel']
        self.noiseType = simuParas['noiseType']
        self.filterType = simuParas['filterType']
        self.signalMode = simuParas['signalMode']
        self.trainMode = simuParas['trainMode']
        self.filterMode = simuParas['filterMode']
        self.selectMode = simuParas['selectMode']
        self.nNodes = simuParas['nNodes']
        self.nClasses = simuParas['nClasses']
        self.N_C = simuParas['N_C']
        self.S = simuParas['S']
        self.graphType = simuParas['graphType']
        self.tMax = simuParas['tMax']
        self.alpha = simuParas['alpha']
        self.nEpochs = simuParas['nEpochs']
        if 'model_number' in simuParas.keys():
            self.model_number = simuParas['model_number']
        else:
            self.model_number = 1
                
        # Graph options
        self.graphOptions = graphOptions
        
        # Model parameters (optional)
        if 'modelParas' in kwargs.keys():
            self.modelParas = kwargs['modelParas']
            self.C = self.modelParas['C']
            self.K = self.modelParas['K']
            self.filterTrainType = self.modelParas['filterTrainType']
        else:
            self.C = self.nNodes
            self.K = 5
            self.filterTrainType = 'g'
            self.modelParas = {}
            self.modelParas['C']= self.C
            self.modelParas['K']= self.K
            self.modelParas['filterTrainType']= self.filterTrainType
        if 'q' in self.modelParas.keys():
            self.q = self.modelParas['q']
        else:
            self.q = 4
            
        # Experiment parameters (optional)
        if 'expParas' in kwargs.keys():
            self.expParas = kwargs['expParas']
            self.nRealiz = self.expParas['nRealiz']
        else:
            self.expParas = {}
            self.nRealiz = 1
            self.expParas['nRealiz'] = self.nRealiz
            
        ## Save settings (optional)
        # Including:
        # thisFilename_SLOG = 'sourceLocSLOGNET'
        # saveDirRoot = 'experiments'
        # saveDir = 'experiments/sourceLocSLOGNET'
        # saveDirRoot_dropbox = '/Users/changye/Dropbox'
        # saveDir_dropbox = '/Users/changye/Dropbox/experiments'
        if 'saveSettings' in kwargs.keys():
            self.saveSettings = kwargs['saveSettings']
            self.thisFilename_SLOG = self.saveSettings['thisFilename_SLOG']
            self.saveDirRoot = self.saveSettings['saveDirRoot'] # Relative location where to save the file
            self.saveDir = self.saveSettings['saveDir']# Dir where to save all the results from each run
            self.saveDirRoot_dropbox = self.saveSettings['saveDirRoot_dropbox']
            self.saveDir_dropbox = self.saveSettings['saveDir_dropbox']         
        else:
            self.thisFilename_SLOG = 'sourceLocSLOGNET'
            self.saveDirRoot = 'experiments' # Relative location where to save the file
            self.saveDir = os.path.join(self.saveDirRoot, self.thisFilename_SLOG) # Dir where to save all the results from each run
            saveDirRoot_dropbox = '/Users/changye'
            saveDirRoot_dropbox = os.path.join(saveDirRoot_dropbox, 'Dropbox')
            self.saveDirRoot_dropbox = os.path.join(saveDirRoot_dropbox, 'onlineResults')
            self.saveDir_dropbox = os.path.join(self.saveDirRoot_dropbox, self.saveDir)
            
            self.saveSettings = {}
            self.saveSettings['thisFilename_SLOG'] = self.thisFilename_SLOG            
            self.saveSettings['saveDirRoot'] = self.saveDirRoot
            self.saveSettings['saveDir'] = self.saveDir
            self.saveSettings['saveDirRoot_dropbox'] = self.saveDirRoot_dropbox
            self.saveSettings['saveDir_dropbox'] = self.saveDir_dropbox
        
        self.experiment_results = []
        for i in range(self.nRealiz):
            result_i = self.run_single_experiment()
            self.experiment_results.append(result_i)
       
    def get_experiment_result(self):
        return self.experiment_results
            
    def run_single_experiment(self,**kwargs):    
        ## kwargs:

        #\\\ Create .txt to store the values of the setting parameters for easier
        # reference  when running multiple experiments
        today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Append date and time of the run to the directory, to avoid several runs of
        # overwritting each other.
        saveDir = self.saveDir + '-' + self.graphType + '-' + today
        saveDir_dropbox = self.saveDir_dropbox + '-' + self.graphType + '-' + today
        # 
        saveDirs = {}
        saveDirs['saveDir'] = saveDir
        saveDirs['saveDir_dropbox'] = saveDir_dropbox

        # Create directory
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        if not os.path.exists(saveDir_dropbox):
            print("Create: ",saveDir_dropbox)
            os.makedirs(saveDir_dropbox)    

        useGPU = True
        if useGPU and torch.cuda.is_available():
            device = 'cuda:0'
            torch.cuda.empty_cache()
        else:
            device = 'cpu'
        # Notify:
        print("Device selected: %s" % device)   

        # Create the file where all the (hyper)parameters are results will be saved.
        varsFile = os.path.join(saveDir,'hyperparameters.txt')
        with open(varsFile, 'w+') as file:
            file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        # Parameter pass
#         nNodes = self.nNodes
#         graphType = self.graphType
#         graphOptions = self.graphOptions
#         simuParas = self.simuParas
#         saveDirs = self.saveDirs
#         nTest = self.nTest
#         nValid = self.nValid
#         nTrain_slog = self.nTrain_slog
#         tMax = self.tMax
#         nClasses = self.nClasses
#         useGPU = self.useGPU
        #\\\ Save values:
        writeVarValues(varsFile, {'nNodes': self.nNodes, 'graphType': self.graphType})
        writeVarValues(varsFile, self.graphOptions)
        writeVarValues(varsFile, self.simuParas)
        writeVarValues(varsFile, self.modelParas)           
        writeVarValues(varsFile, saveDirs)
        writeVarValues(varsFile, {'nTrain_slog': self.nTrain_slog,
                                  'nValid': self.nValid,
                                  'nTest': self.nTest,
                                  'tMax': self.tMax,
                                  'nClasses': self.nClasses,
                                  'useGPU': useGPU})


        # Create the file where all the (hyper)parameters are results will be saved.
        varsFile = os.path.join(saveDir_dropbox,'hyperparameters.txt')
        with open(varsFile, 'w+') as file:
            file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

        #\\\ Save values:
        writeVarValues(varsFile, {'nNodes': self.nNodes, 'graphType': self.graphType})
        writeVarValues(varsFile, self.graphOptions)
        writeVarValues(varsFile, self.simuParas)
        writeVarValues(varsFile, self.modelParas)        
        writeVarValues(varsFile, saveDirs)
        writeVarValues(varsFile, {'nTrain': self.nTest,
                                  'nValid': self.nValid,
                                  'nTest': self.nTest,
                                  'tMax': self.tMax,
                                  'nClasses': self.nClasses,
                                  'useGPU': useGPU})
        
        if 'optimAlg' in kwargs.keys():
            optimAlg = kwargs['optimAlg']
        else:
            optimAlg = 'ADAM'
            
        if 'learningRate' in kwargs.keys():
            learningRate = kwargs['learningRate']
        else:
            learningRate = 0.01
            
        if 'beta1' in kwargs.keys():
            beta1 = kwargs['beta1']
        else:
            beta1 = 0.9
            
        if 'beta2' in kwargs.keys():
            beta2 = kwargs['beta2']
        else:
            beta2 = 0.999
            
        ## Graph generation
        G = SLOGtools.Graph(self.graphType, self.nNodes, self.graphOptions, save_dir = saveDir, save_dir_dropBox = saveDir_dropbox)
        G.computeGFT()
        sourceNodes, communityLabels,communityNodeList = SLOGtools.computeSourceNodes_slog(G.A, self.nClasses, self.N_C, mode = 'random', save_dir = saveDir, save_dir_dropBox = saveDir_dropbox)

        d_slog,An_slog, eigenvalues_slog, V_slog   = SLOGtools.get_eig_normalized_adj(G.A)
        gso = An_slog
        # Write community information
        communityInfo = {}
        communityInfo['sourceNodes'] = sourceNodes
        communityInfo['communityLabels'] = communityLabels
        communityInfo['communityNodeList'] = communityNodeList  
        graphInfo = {}
        graphInfo['gso'] = gso
        graphInfo['eigenvalues'] = eigenvalues_slog
        graphInfo['degrees'] = d_slog
        writeVarValues(varsFile, communityInfo)                  
        writeVarValues(varsFile, graphInfo)         
        ## Data generation
#         data = SLOGdata.SLOG_ClassificationData(G, self.nTrain_slog, self.nValid, self.nTest, sourceNodes, communityNodeList, communityLabels, V_slog, eigenvalues_slog, L = self.L, tMax = self.tMax, alpha = self.alpha, filterMode = self.filterMode, selectMode = self.selectMode, signalMode = self.signalMode, filterType = self.filterType, noiseLevel = self.noiseLevel)
#         data = SLOGdata.SLOG_GeneralData(G, self.nTrain_slog, self.nValid, self.nTest, self.S, V_slog, eigenvalues_slog, L = self.L, alpha = self.alpha,filterType = self.filterType, noiseLevel = self.noiseLevel, noiseType = self.noiseType)
        data = SLOGdata.SLOG_ClassificationData_v2(gso, self.nTrain_slog, self.nValid, self.nTest, sourceNodes, communityNodeList, communityLabels, V_slog, eigenvalues_slog, L = self.L, tMax = self.tMax, alpha = self.alpha, filterMode = self.filterMode, selectMode = self.selectMode, signalMode = self.signalMode, filterType = self.filterType, noiseLevel = self.noiseLevel)
        data.astype(torch.float64)
        data.expandDims()
        
        C = self.C
        K = self.K
        filterTrainType = self.filterTrainType #'g'
        thisLoss = SLOGtools.myLoss
        thisEvaluator = SLOGevaluator.evaluate
        
        if self.model_number == 1:
            thisObject = SLOGobj.myFunction_slog_3
            SLOG_net = SLOGarchi.GraphSLoG_v3(V_slog,self.nNodes,self.q,self.K, thisObject)        
        else:
            thisObject = SLOGobj.myFunction_slog_1
            SLOG_net = SLOGarchi.GraphSLoG_v1(V_slog,self.nNodes,self.C,self.K, thisObject)

        # thisObject = SLOGobj.myFunction_slog_2
        # SLOG_net = SLOGarchi.GraphSLoG_v2(V_slog,nNodes,C,K, thisObject)


        model_name = 'SLOG-Net'

        thisOptim = optim.Adam(SLOG_net.parameters(), lr = learningRate, betas = (beta1,beta2))
        thisTrainer = SLOGtrainer.slog_Trainer

        myModel = SLOGmodel.Model(SLOG_net,thisLoss,thisOptim, thisTrainer,thisEvaluator,device, model_name,  saveDir, saveDir_dropbox = saveDir_dropbox)

        result_train = myModel.train(data,self.nEpochs, self.batchsize_slog, validationInterval = 40,trainMode = self.trainMode,tMax = self.tMax, filterTrainType = self.filterTrainType) # model, data, nEpochs, batchSize
        
        best_model = result_train['bestModel']
        minLossValid = result_train['minLossValid']
        minLossTrain = result_train['minLossTrain']
        
        writeVarValues(varsFile, result_train)
            
        results = {}
        results['model'] = myModel
        results['training result'] = result_train
        
        #####################################################################
        ################# CrsGNN ############################################
        #####################################################################
        
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
        hParamsSelGNN['dimLayersMLP'] = [self.nClasses] # Dimension of the fully connected layers after the GCN layers
        hParamsCrsGNN = deepcopy(hParamsSelGNN)
        hParamsCrsGNN['name'] = 'CrsGNN'
        hParamsCrsGNN['rho'] = nn.MaxPool1d
        hParamsCrsGNN['order'] = None # We don't need any special ordering, since
        # it will be determined by the hierarchical clustering algorithm
        writeVarValues(varsFile, hParamsCrsGNN)
        
        printInterval = 0 # After how many training steps, print the partial results
            # if 0 never print training partial results.
        xAxisMultiplierTrain = 100 # How many training steps in between those shown in
            # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
        xAxisMultiplierValid = 10 # How many validation steps in between those shown,
            # same as above.
        figSize = 5 # Overall size of the figure that contains the plot
        lineWidth = 2 # Width of the plot lines
        markerShape = 'o' # Shape of the markers
        markerSize = 3 # Size of the markers
        
        trainingOptions = {}
        validationInterval = 40
        
        trainingOptions['saveDir'] = saveDir
        trainingOptions['saveDir_dropbox'] = saveDir_dropbox       
        trainingOptions['printInterval'] = printInterval
        trainingOptions['validationInterval'] = validationInterval
        writeVarValues(varsFile, trainingOptions)
        
        thisName = hParamsCrsGNN['name']

        # Save seed
        #   PyTorch seeds
        torchState = torch.get_rng_state()
        torchSeed = torch.initial_seed()
        #   Numpy seeds
        numpyState = np.random.RandomState().get_state()
        #   Collect all random states
        randomStates = []
        randomStates.append({})
        randomStates[0]['module'] = 'numpy'
        randomStates[0]['state'] = numpyState
        randomStates.append({})
        randomStates[1]['module'] = 'torch'
        randomStates[1]['state'] = torchState
        randomStates[1]['seed'] = torchSeed
        #   This list and dictionary follows the format to then be loaded, if needed,
        #   by calling the loadSeed function in Utils.miscTools
        SLOGtools.saveSeed(randomStates, saveDir)
        SLOGtools.saveSeed(randomStates, saveDir_dropbox)
        
        #\\\ Architecture
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
                                 coarsening = True)
        coarsened_outputs_save_name = 'CrsGNNcoarsen-' + self.graphType + '.npy'
        coarsened_outputs_save_dir = os.path.join(saveDir, coarsened_outputs_save_name) 
        torch.save(thisArchit.coarsened_outputs, coarsened_outputs_save_dir)
        coarsened_outputs_save_dir_dropBox = os.path.join(saveDir_dropbox, coarsened_outputs_save_name) 
        torch.save(thisArchit.coarsened_outputs, coarsened_outputs_save_dir_dropBox)
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
        thisArchit.to(device)

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
                     saveDir_dropbox = saveDir_dropbox)
        
        print("Training model %s..." % CrsGNN, end = ' ', flush = True)
        #Train
        print()
        batchSize = self.batchsize_slog
        thisTrainVars = CrsGNN.train(data, self.nEpochs, batchSize, **trainingOptions)
        # Save the variables
        thisModel = 'CrsGNN'
        lossTrain = thisTrainVars['lossTrain']
        costTrain = thisTrainVars['costTrain']
        lossValid = thisTrainVars['lossValid']
        costValid = thisTrainVars['costValid']
        print('CrsGNN:')
        print('lossTrain = ',lossTrain)
        print('costTrain = ',costTrain)
        print('lossValid = ',lossValid)
        print('costValid = ',costValid)
        results['CrsGNNmodel'] = CrsGNN
        results['lossTrain'] = lossTrain
        results['costTrain'] = costTrain
        results['lossValid'] = lossValid
        results['costValid'] = costValid   
        writeVarValues(varsFile, results)
        return results
    
