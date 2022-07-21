# 2018/10/02~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
model.py Model Module

Utilities useful for working on the model

Model: binds together the architecture, the loss function, the optimizer,
       the trainer, and the evaluator.
"""

import os
import torch

class Model:
    """
    Model: binds together the architecture, the loss function, the optimizer,
        the trainer, and the evaluator.
        
    Initialization:
        
        architecture (nn.Module)
        loss (nn.modules.loss._Loss)
        optimizer (nn.optim)
        trainer (Modules.training)
        evaluator (Modules.evaluation)
        device (string or device)
        name (string)
        saveDir (string or path)
        
    .train(data, nEpochs, batchSize, **kwargs): train the model for nEpochs 
        epochs, using batches of size batchSize and running over data data 
        class; see the specific selected trainer for extra options
    
    .evaluate(data): evaluate the model over data data class; see the specific
        selected evaluator for extra options
        
    .save(label = '', [saveDir=dirPath]): save the model parameters under the
        name given by label, if the saveDir is different from the one specified
        in the initialization, it needs to be specified now
        
    .load(label = '', [loadFiles=(architLoadFile, optimLoadFile)]): loads the
        model parameters under the specified name inside the specific saveDir,
        unless they are provided externally through the keyword 'loadFiles'.
        
    .getTrainingOptions(): get a dict with the options used during training; it
        returns None if it hasn't been trained yet.'
    """
    
    def __init__(self,
                 # Architecture (nn.Module)
                 architecture,
                 # Loss Function (nn.modules.loss._Loss)
                 loss,
                 # Optimization Algorithm (nn.optim)
                 optimizer,
                 # Training Algorithm (Modules.training)
                 trainer,
                 # Evaluating Algorithm (Modules.evaluation)
                 evaluator,
                 # Other
                 device, name, saveDir, **kwargs):
        
        #\\\ ARCHITECTURE
        # Store
        self.archit = architecture
        # Move it to device
#         self.archit.to(device)
        # Count parameters (doesn't work for EdgeVarying)
        self.nParameters = 0
        for param in list(self.archit.parameters()):
            if len(param.shape)>0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.nParameters += thisNParam
            else:
                pass
        #\\\ LOSS FUNCTION
        self.loss = loss
        #\\\ OPTIMIZATION ALGORITHM
        self.optim = optimizer
        #\\\ TRAINING ALGORITHM
        self.trainer = trainer
        #\\\ EVALUATING ALGORITHM
        self.evaluator = evaluator
        #\\\ OTHER
        # Device
        self.device = device
        # Model name
        self.name = name
        # Saving directory
        self.saveDir = saveDir
        if 'saveDir_dropbox' in kwargs.keys():
            self.saveDir_dropbox = kwargs['saveDir_dropbox']
            self.saveTodropbox = True
            print('Model is saving to dropbox:', self.saveDir_dropbox)            
        else:
            self.saveDir_dropbox = None
            self.saveTodropbox = False    
        
    def train(self, data, nEpochs, batchSize, **kwargs):
        
        self.trainer = self.trainer(self, data, nEpochs, batchSize, **kwargs)
#         self.trainer = self.trainer(data, nEpochs, batchSize, **kwargs)        
        
        return self.trainer.train()
    
    def evaluate(self, data, **kwargs):
        
        return self.evaluator(self, data, **kwargs)
    
    def save(self, label = '', **kwargs):
        if 'saveDir' in kwargs.keys():
            saveDir = kwargs['saveDir']
        else:
            saveDir = self.saveDir
        if 'saveDir_dropbox' in kwargs.keys():
            saveDir_dropbox = kwargs['saveDir_dropbox']
            saveTodropbox = True
        elif self.saveTodropbox == True:
            saveDir_dropbox = self.saveDir_dropbox
            saveTodropbox = True
        else:
            saveDir_dropbox = self.saveDir_dropbox
            saveTodropbox = self.saveTodropbox
        saveModelDir = os.path.join(saveDir,'savedModels')

        # Create directory savedModels if it doesn't exist yet:
        if not os.path.exists(saveModelDir):
            os.makedirs(saveModelDir)            
        saveFile = os.path.join(saveModelDir, self.name) 
#         print('Saved to', saveFile+'Archit'+ label+'.ckpt')
        torch.save(self.archit.state_dict(), saveFile+'Archit'+ label+'.ckpt')
        torch.save(self.optim.state_dict(), saveFile+'Optim'+label+'.ckpt')
        if saveTodropbox == True:
            saveModelDir_dropbox = os.path.join(saveDir_dropbox,'savedModels')           
            if not os.path.exists(saveModelDir_dropbox):
                os.makedirs(saveModelDir_dropbox) 
            saveFile_dropbox = os.path.join(saveModelDir_dropbox, self.name)       
            torch.save(self.archit.state_dict(), saveFile_dropbox+'Archit'+ label+'.ckpt')
            torch.save(self.optim.state_dict(), saveFile_dropbox+'Optim'+label+'.ckpt')
        

    def load(self, label = '', **kwargs):
        if 'loadFiles' in kwargs.keys():
            (architLoadFile, optimLoadFile) = kwargs['loadFiles']
        else:
            saveModelDir = os.path.join(self.saveDir,'savedModels')
            architLoadFile = os.path.join(saveModelDir,
                                          self.name + 'Archit' + label +'.ckpt')
            optimLoadFile = os.path.join(saveModelDir,
                                         self.name + 'Optim' + label + '.ckpt')
        self.archit.load_state_dict(torch.load(architLoadFile))
        self.optim.load_state_dict(torch.load(optimLoadFile))
        
    def load_from_dropBox(self, modelDir, label = ''):
        saveModelDir = os.path.join(modelDir,'savedModels')
        architFileName = self.name + 'Archit' + label +'.ckpt'
        optimFileName = self.name + 'Optim' + label + '.ckpt'
        architLoadFile = os.path.join(saveModelDir, architFileName )
        optimLoadFile = os.path.join(saveModelDir, optimFileName )
        self.archit.load_state_dict(torch.load(architLoadFile))
        self.optim.load_state_dict(torch.load(optimLoadFile))         

    def getTrainingOptions(self):
        
        return self.trainer.trainingOptions \
                        if 'trainingOptions' in dir(self.trainer) \
                    else None

    def __repr__(self):
        reprString  = "Name: %s\n" % (self.name)
        reprString += "Number of learnable parameters: %d\n"%(self.nParameters)
        reprString += "\n"
        reprString += "Model architecture:\n"
        reprString += "----- -------------\n"
        reprString += "\n"
        reprString += repr(self.archit) + "\n"
        reprString += "\n"
        reprString += "Loss function:\n"
        reprString += "---- ---------\n"
        reprString += "\n"
        reprString += repr(self.loss) + "\n"
        reprString += "\n"
        reprString += "Optimizer:\n"
        reprString += "----------\n"
        reprString += "\n"
        reprString += repr(self.optim) + "\n"
        reprString += "Training algorithm:\n"
        reprString += "-------- ----------\n"
        reprString += "\n"
        reprString += repr(self.trainer) + "\n"
        reprString += "Evaluation algorithm:\n"
        reprString += "---------- ----------\n"
        reprString += "\n"
        reprString += repr(self.evaluator) + "\n"
        return reprString
