from SLOGmodules import SLOGpipeline as SLOGpipeline
import os
from SLOGmodules import SLOGSaveDir as SLOGSaveDir

### Save settings
location = 'home'
# location = 'office'
saveDir_result = SLOGSaveDir.get_save_settings(location)
saveDir_dropbox = saveDir_result['saveDir_dropbox']
saveSettings = saveDir_result['saveSettings']

### Model parameters
K = 5 # number of layers

# filterTrainType = 'h'
filterTrainType = 'wt'

### Simulation parameters
simuParas = {}
nTrain_slog = 200000
batchsize_slog = 400
nValid = batchsize_slog
nTest = batchsize_slog
nEpochs = 50
# model_number = 0
model_number = 1 # SLOG-Net-v3 

### Data parameters
L = 5
alpha = 1.0

### Graph parameters

nNodes = 20 # Number of nodes
nClasses = 4 # Number of classes (i.e. number of communities)
N_C = 2 # Number of sources per signal
C = nNodes # constrain constant
q = 4 # Number of constrain vectors for SLOG-Net-v3 
# nNodes = 100 # Number of nodes
# nClasses = 5 # Number of classes (i.e. number of communities)
# N_C = 10 # Number of sources per signal

simuParas['nNodes'] = nNodes
simuParas['nClasses'] = nClasses
simuParas['N_C'] = N_C
simuParas['S'] = N_C
simuParas['model_number'] = model_number

graphType = 'SBM' # Type of graph
graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
graphOptions['nCommunities'] = nClasses # Number of communities
graphOptions['probIntra'] = 0.8 # Probability of drawing edges intra communities
graphOptions['probInter'] = 0.2 # Probability of drawing edges inter communities

# graphType = 'Random Geometric' # Type of graph
# graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
# graphOptions['distance'] = 0.2 # Number of communities

# graphType = 'BA' # Type of graph
# graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
# graphOptions['alpha'] = 1.0

# graphType = 'ER' # Type of graph
# graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
# graphOptions['probIntra'] = 0.3 # Probability of drawing edges

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
# filterMode = 'default'
filterMode = 'Wt'

## Selection mode: random or nodes with top-N_C degree
selectMode = 'random'

## Noise level
noiseLevel = 0.0
noiseType = 'uniform'

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


tMax = None # Maximum number of diffusion times (W^t for t < tMax)
# tMax = nNodes
tMax = 5
simuParas['tMax'] = tMax

## model parameters
modelParas = {}
modelParas['filterTrainType'] = filterTrainType
modelParas['C'] = C
modelParas['q'] = q
modelParas['K'] = K

# thisFilename_SLOG = 'sourceLocSLOGNET'
# saveDirRoot = 'experiments' # Relative location where to save the file
# saveDir = os.path.join(saveDirRoot, thisFilename_SLOG) # Dir where to save all the results from each run
# saveSettings = {}
# saveSettings['thisFilename_SLOG'] = thisFilename_SLOG         
# saveSettings['saveDirRoot'] = saveDirRoot # Relative location where to save the file
# saveSettings['saveDir'] = os.path.join(saveDirRoot, thisFilename_SLOG) # Dir where to save all the results from each run
# saveDirRoot_dropbox = r"C:\Users\Chang Ye\Dropbox\onlineresults"
# saveSettings['saveDirRoot_dropbox'] = saveDirRoot_dropbox
# saveSettings['saveDir_dropbox'] = os.path.join(saveDirRoot_dropbox, saveDir)

## Experiment parameters
expParas = {}
expParas['nRealiz'] = 10

simuParas['noiseLevel'] = 0.0
simuParas['model_number'] = 1
modelParas['q'] = 4
modelParas['filterTrainType'] = 'wt'
result_1 = SLOGpipeline.slog_classification_experiments(simuParas = simuParas, 
                 graphOptions = graphOptions, 
                 modelParas = modelParas, expParas = expParas, saveSettings = saveSettings)

