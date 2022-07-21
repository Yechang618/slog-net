# -*- coding: utf-8 -*-
"""
Created on Sun May 08 22:56:00 2022

@author: Chang Ye

"""

import torch
import torch.nn as nn
import math
import numpy as np
from scipy import linalg
from numpy import linalg as LA
from torch.autograd import Variable
import alegnn.utils.graphTools as graph
from sklearn.cluster import SpectralClustering

def fast_inverse(A,rho):
    N = A.shape[0]
    a = torch.diagonal(A, 0)
    sum_a = sum(a)
    a_1 = (1/a).reshape(N,1)
    A_1 = torch.diag(a_1.reshape(N))
#     return A_1 - torch.matmul(torch.matmul(A_1,torch.ones(N,N)),A_1)*(rho/(1+rho*sum_a))
    return A_1 - (rho/(1+rho*sum_a))*torch.matmul(a_1,a_1.transpose(0,1))
                                                  
def min_RE(x_pre,x_ture):
    re_1 = LA.norm(x_pre - x_ture)/LA.norm(x_ture)
    re_2 = LA.norm(x_pre + x_ture)/LA.norm(x_ture)
    RE = min(re_1,re_2)
    sign = re_1 < re_2 
    return RE,sign
def generate_V(N):
    V = torch.randn([N,N])
    Vout = torch.zeros([N,N])
    Vout[0,:] = V[0,:]/torch.norm(V[0,:]);
    for i in range(1,N):
        Vout[i,:] = V[i,:]
        for j in range(0,i):
            Vout[i,:] = Vout[i,:] - torch.dot(Vout[i,:],Vout[j,:])*Vout[j,:];
        Vout[i,:] = Vout[i,:]/torch.norm(Vout[i,:]);
    return Vout

def X_generate(N,P,S):
## Generating full rank X
    miu = 0
    sigma = 0.5
    R = abs(np.random.normal(miu,sigma,[S,P]))
    sign = (np.random.binomial(1,0.5,[S,P])-0.5)*2
    X = np.multiply(R,sign)
    X0 = np.zeros([N-S, P])
    X = np.concatenate([X, X0]) # Pad zeros to generate the full signals 
    for p in range(0,P):
        temp_X = X[:,p]
#         idx = np.arange(N)
        Randindex =  np.random.permutation(N)
        X[:,p] = temp_X[Randindex]
    return X

def softshrink(x, lambd):
    mask1 = x > lambd
    mask2 = x < -lambd
    out = torch.zeros_like(x)
    out += mask1.float() * -lambd + mask1.float() * x
    out += mask2.float() * lambd + mask2.float() * x
    return out

################################
############# ADMM #############
################################

def admm(Y,V,rho_0,eta_0,C,N_ite,max_re):
    N = V.shape[0]
    P = Y.shape[1]
#     Y = torch.tensor(Y)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    v = torch.tensor(np.zeros(N))
    x = torch.tensor(np.zeros(N*P))
    u = torch.tensor(np.zeros(N*P))
    eta = torch.tensor(np.zeros(1))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    n_ite = 0
    max_re_matched = 0
    while n_ite < N_ite and max_re_matched ==0 :
        v_old = v
        x_old = x
        ZIk_inv = fast_inverse(torch.matmul(torch.transpose(Z,0,1),Z),eta_0/rho_0)/rho_0
#         ZIk = rho_0*torch.matmul(torch.transpose(Z,0,1),Z) + eta_0*II
#         ZIk_inv = torch.pinverse(ZIk)
        v_temp = torch.matmul(torch.transpose(Z,0,1),rho_0*x - u) + (eta_0*C-eta)*In
        v = torch.matmul(ZIk_inv, v_temp)
        X_update = torch.nn.Softshrink(lambd = 1/rho_0)
        x = X_update(torch.matmul(Z,v) + 1/rho_0*u)
        u = u + rho_0*(torch.matmul(Z,v) - x)
        eta = eta + eta_0*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C)
        re = ((v-v_old)**2).mean()/(1e-10 + ((v_old)**2).mean())
        if re < max_re**2:
#             print('re = ',re)
            max_re_matched = 1
        n_ite += 1
    return x,v,n_ite,max_re_matched

def Ladmm(Y,V,rho_0,C,N_ite,max_re,alpha, beta,max_v_ite):
    eta_0 = rho_0
    N = V.shape[0]
    P = Y.shape[1]
#     Y = torch.tensor(Y)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    v = torch.tensor(np.zeros(N))
    x = torch.tensor(np.zeros(N*P))
    u = torch.tensor(np.zeros(N*P))
    eta = torch.tensor(np.zeros(1))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    n_ite = 0
    max_re_matched = 0
    while n_ite < N_ite and max_re_matched ==0 :
        v_old = v
        x_old = x
#         ZIk = rho_0*torch.matmul(torch.transpose(Z,0,1),Z) + eta_0*II
#         ZIk_inv = torch.pinverse(ZIk)
#         v_temp = torch.matmul(torch.transpose(Z,0,1),rho_0*x - u) + (eta_0*C-eta)*In
#         v = torch.matmul(ZIk_inv, v_temp)
        v_cost_old = torch.norm(torch.matmul(Z,v_old) + u/rho_0 - x) + torch.norm( torch.matmul(torch.tensor(np.ones([1,N])),v_old) - C + eta/rho_0 )
        v_cost_new = 10*v_cost_old
        v_gd = torch.matmul(torch.transpose(Z,0,1),torch.matmul(Z,v_old) + u/rho_0 - x)
        v_gd = v_gd + ( torch.matmul(torch.tensor(np.ones([1,N])),v) - C + eta/rho_0 )*In
        step_size = alpha
        v_ite = 0
        while v_cost_new >= v_cost_old and v_ite < max_v_ite:
            v_ite += 1
            v = v_old - step_size*v_gd
            v_cost_new = torch.norm(torch.matmul(Z,v) + u/rho_0 - x) + torch.norm( torch.matmul(torch.tensor(np.ones([1,N])),v) - C + eta/rho_0 )
            step_size = step_size*beta
#             print([step_size,v_cost_new,v_cost_old])
        X_update = torch.nn.Softshrink(lambd = 1/rho_0)
        x = X_update(torch.matmul(Z,v) + 1/rho_0*u)
        u = u + rho_0*(torch.matmul(Z,v) - x)
        eta = eta + eta_0*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C)
        re = ((v-v_old)**2).mean()/(1e-10 + ((v_old)**2).mean())
        if re < max_re**2:
#             print('re = ',re)
            max_re_matched = 1
        n_ite += 1
    return x,v,n_ite,max_re_matched
################################
####### DATA TOOLS #############
################################
def X_convert(X,sourceNodes,N):
    # X should be N times P
    N_sourceNode = len(sourceNodes)
    P = X.shape[0]
    X_out = torch.zeros(N,P)
    for p in range(P):
        print(sourceNodes[X[p]])
        n = sourceNodes[X[p]]
        X_out[n][p] = 1
    return X_out

################################
####### GRAPH TOOLS ############
################################

def computeSourceNodes_slog(A, C, N_C, mode = 'default'):
    """
    computeSourceNodes: compute source nodes for the source localization problem
    
    Input:
        A (np.array): adjacency matrix of shape N x N
        C (int): number of classes
        
    Output:
        sourceNodes (list): contains the indices of the C source nodes
        
    Uses the adjacency matrix to compute C communities by means of spectral 
    clustering, and then selects the node with largest degree within each 
    community
    """
    sourceNodes = []
    degree = np.sum(A, axis = 0) # degree of each vector
    # Compute communities
    communityClusters = SpectralClustering(n_clusters = C,
                                           affinity = 'precomputed',
                                           assign_labels = 'discretize')
    communityClusters = communityClusters.fit(A)
    communityLabels = communityClusters.labels_
    communityList = []
    # For each community
    for c in range(C):
        communityNodes = np.nonzero(communityLabels == c)[0]
        degreeSorted = np.argsort(degree[communityNodes])
        if mode == 'random':
            np.random.shuffle(degreeSorted)
        sourceNodes.append(communityNodes[degreeSorted[-N_C:]])
        communityList.append(communityNodes)
    return sourceNodes, communityLabels,communityList

################################
####### DATA GENERATION ########
################################
def data_generate(N,P,V, theta,alpha):
#     N = 10; theta = 0.3; N_realiz = 1;  # X parameters
#     S = int(N*theta); P = 2;
#     alpha = 0.1; # Graph & Filter parameters
    # SNR = 0.01;p = 0.2; L = 5;
    # V = myModules.generate_V(N)
    g = np.ones(N)+alpha*np.random.uniform(0,1,N)
    g = N*g/sum(g)
    h = 1./g;# h is h tilde here.
    H = np.dot(V,np.dot(np.diag(h),np.transpose(V)))
    X = X_generate(N,P,S)
    Y = np.dot(H,X)
    Z = linalg.khatri_rao(np.dot(np.transpose(Y),V),V)
    return X,g,Z,Y

def Xdata_generate(N,P,V,g, theta,alpha):
    S = int(N*theta); #P = 2;
    h = 1./g;# h is h tilde here.
    H = np.dot(V,np.dot(np.diag(h),np.transpose(V)))
    X = X_generate(N,P,S)
    Y = np.dot(H,X)
    return X,Y

def Xsdata_generate(N,P,V,g, theta,alpha):
    S = int(N*theta); #P = 2;
    h = 1./g;# h is h tilde here.
    H = np.dot(V,np.dot(np.diag(h),np.transpose(V)))
    X = X_generate(N,P,S)
    Y = np.dot(H,X)
    return X,Y

def Xdata_generate_v2(N,P,V,gs, theta,alpha):
    hs = 1./gs;# h is h tilde here.
    X = X_generate(N,P,S)
    Y = []
    for p in range(P):
        x = X[:,p]
        h_p = hs[:,p] 
        H = np.dot(V,np.dot(np.diag(h_p),np.transpose(V)))
        y = np.dot(H,x)
        y = y.reshape([N,1])
        if p == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y), axis=1)
    return X,Y

def g_generate(N,P,alpha):
    gs = np.ones([N,P])+alpha*np.random.uniform(0,1,[N,P])
    for p in range(P):
        gs[:,p] = N*gs[:,p]/sum(gs[:,p])
    return gs

def g_generate_gso(N,alpha, eigenvalues,L):
    Vd = np.vander(eigenvalues,L)
    Vd = np.fliplr(Vd)
    g = alpha*np.random.normal(0, 1, L)
    g[0] = 1
    g_tilde = np.dot(Vd,g)
    g_tilde = N*g_tilde/sum(g_tilde)
    return g_tilde

def h_generate_gso(N,alpha, eigenvalues,L):
    Vd = np.vander(eigenvalues,L)
    Vd = np.fliplr(Vd)
    h = alpha*np.random.normal(0, 1, L)
    h[0] = 1
    h_tilde = np.dot(Vd,h)
    h_tilde = N*h_tilde/sum(h_tilde)
    g_tilde = 1/h_tilde    
    return g_tilde

def g_batch_generate(N,nBatches,alpha):
    g_batch = np.ones([N,nBatches])+alpha*np.random.normal(0, 1, [N,nBatches])
    for p in range(nBatches):
        g_batch[:,p] = N*g_batch[:,p]/sum(g_batch[:,p])
    return g_batch


def g_batch_generate_gso(N,nBatches,alpha, eigenvalues,L):
    Vd = np.vander(eigenvalues,L)
    Vd = np.fliplr(Vd)
    g = alpha*np.random.normal(0, 1, [L,nBatches])
    e1 = np.zeros([L,nBatches])
    e1[0,:] = 1
    g = g + e1
    g_batch = np.dot(Vd,g)
    for p in range(nBatches):
#         g_batch[:,p] = N*g_batch[:,p]/LA.norm(g_batch[:,p],1)
        g_batch[:,p] = N*g_batch[:,p]/sum(g_batch[:,p])
    return g_batch

def h_batch_generate_gso(N,nBatches,alpha, eigenvalues,L):
    Vd = np.vander(eigenvalues,L)
    Vd = np.fliplr(Vd)
    h = alpha*np.random.normal(0, 1, [L,nBatches])
    e1 = np.zeros([L,nBatches])
    e1[0,:] = 1
    h = h + e1
    h_batch = np.dot(Vd,h)
    g_batch = 1/h_batch
    for p in range(nBatches):
        g_batch[:,p] = N*g_batch[:,p]/sum(g_batch[:,p])
    return g_batch

def generate_normalized_gso_laplaciant(N,p):
    connected_nonperm = 0
    temp_count_perm_pairs = 10
    while temp_count_perm_pairs > 1e-10:
        temp_connect = 0
        while temp_connect < 1e-10:
            random = np.random.random((N, N))
            tri = np.tri(N, k=-1)
            # initialize adjacency matrix
            gso = np.zeros((N, N))
            # assign intra-community edges
            gso[np.logical_and(tri, random < p)] = 1
            gso += gso.T
            # connected check
            d = np.sum(gso, axis=1)
            temp_di_mutiply = 1
            for i in range(N):
                temp_di_mutiply = temp_di_mutiply*d[i]
            if abs(temp_di_mutiply) > 1e-10:
                temp_connect = 1
        Lp = np.diag(d) - gso
        Lpn = np.zeros((N,N))
        An = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                Lpn[i,j] = Lp[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
                An[i,j] = gso[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
        eigenvalues, V = np.linalg.eig(An)        
        # Permutation check
        temp_count_perm_pairs = 0
        for k in range(N):
            VV = np.outer(V[:,k],V[:,k])
            temp_non_zeros_entry_count = 0
            for l1 in range(N):
                for l2 in range(N):
                    if abs(VV[l1,l2]) > 1e-5:
                        temp_non_zeros_entry_count += 1
            if temp_non_zeros_entry_count < 5:
                temp_count_perm_pairs += 1
    return gso, d,Lpn, eigenvalues, V  

######################### import from gnn master ##############
def normalizeData(x, ax):
    """
    normalizeData(x, ax): normalize data x (subtract mean and divide by standard 
    deviation) along the specified axis ax
    """
    
    thisShape = x.shape # get the shape
    assert ax < len(thisShape) # check that the axis that we want to normalize
        # is there
    dataType = type(x) # get data type so that we don't have to convert

    if 'numpy' in repr(dataType):

        # Compute the statistics
        xMean = np.mean(x, axis = ax)
        xDev = np.std(x, axis = ax)
        # Add back the dimension we just took out
        xMean = np.expand_dims(xMean, ax)
        xDev = np.expand_dims(xDev, ax)

    elif 'torch' in repr(dataType):

        # Compute the statistics
        xMean = torch.mean(x, dim = ax)
        xDev = torch.std(x, dim = ax)
        # Add back the dimension we just took out
        xMean = xMean.unsqueeze(ax)
        xDev = xDev.unsqueeze(ax)

    # Subtract mean and divide by standard deviation
    x = (x - xMean) / xDev

    return x

def changeDataType(x, dataType):
    """
    changeDataType(x, dataType): change the dataType of variable x into dataType
    """
    
    # So this is the thing: To change data type it depends on both, what dtype
    # the variable already is, and what dtype we want to make it.
    # Torch changes type by .type(), but numpy by .astype()
    # If we have already a torch defined, and we apply a torch.tensor() to it,
    # then there will be warnings because of gradient accounting.
    
    # All of these facts make changing types considerably cumbersome. So we
    # create a function that just changes type and handles all this issues
    # inside.
    
    # If we can't recognize the type, we just make everything numpy.
    
    # Check if the variable has an argument called 'dtype' so that we can now
    # what type of data type the variable is
    if 'dtype' in dir(x):
        varType = x.dtype
    
    # So, let's start assuming we want to convert to numpy
    if 'numpy' in repr(dataType):
        # Then, the variable con be torch, in which case we move it to cpu, to
        # numpy, and convert it to the right type.
        if 'torch' in repr(varType):
            x = x.cpu().numpy().astype(dataType)
        # Or it could be numpy, in which case we just use .astype
        elif 'numpy' in repr(type(x)):
            x = x.astype(dataType)
    # Now, we want to convert to torch
    elif 'torch' in repr(dataType):
        # If the variable is torch in itself
        if 'torch' in repr(varType):
            x = x.type(dataType)
        # But, if it's numpy
        elif 'numpy' in repr(type(x)):
            x = torch.tensor(x, dtype = dataType)
            
    # This only converts between numpy and torch. Any other thing is ignored
    return x

def invertTensorEW(x):
    
    # Elementwise inversion of a tensor where the 0 elements are kept as zero.
    # Warning: Creates a copy of the tensor
    xInv = x.copy() # Copy the matrix to invert
    # Replace zeros for ones.
    xInv[x < zeroTolerance] = 1. # Replace zeros for ones
    xInv = 1./xInv # Now we can invert safely
    xInv[x < zeroTolerance] = 0. # Put back the zeros
    
    return xInv
class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), expandDims(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    
    # All the signals are always assumed to be graph signals that are written
    #   nDataPoints (x nFeatures) x nNodes
    # If we have one feature, we have the expandDims() that adds a x1 so that
    # it can be readily processed by architectures/functions that always assume
    # a 3-dimensional signal.
    
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['targets'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['targets'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['targets'] = None
        
    def getSamples(self, samplesType, *args):
        # samplesType: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['targets']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # Select the corresponding samples
                xSelected = x[selectedIndices]
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xSelected = x[args[0]]
                # And assign the labels
                y = y[args[0]]
                
            # If we only selected a single element, then the nDataPoints dim
            # has been left out. So if we have less dimensions, we have to
            # put it back
            if len(xSelected.shape) < len(x.shape):
                if 'torch' in self.dataType:
                    x = xSelected.unsqueeze(0)
                else:
                    x = np.expand_dims(xSelected, axis = 0)
            else:
                x = xSelected

        return x, y
    
    def expandDims(self):
        
        # For each data set partition
        for key in self.samples.keys():
            # If there's something in them
            if self.samples[key]['signals'] is not None:
                # And if it has only two dimensions
                #   (shape: nDataPoints x nNodes)
                if len(self.samples[key]['signals'].shape) == 2:
                    # Then add a third dimension in between so that it ends
                    # up with shape
                    #   nDataPoints x 1 x nNodes
                    # and it respects the 3-dimensional format that is taken
                    # by many of the processing functions
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(1)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 1)
                elif len(self.samples[key]['signals'].shape) == 3:
                    if 'torch' in repr(self.dataType):
                        self.samples[key]['signals'] = \
                                       self.samples[key]['signals'].unsqueeze(2)
                    else:
                        self.samples[key]['signals'] = np.expand_dims(
                                                   self.samples[key]['signals'],
                                                   axis = 2)
        
    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        
        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers also after conversion. 
        # To do this we need to match the desired dataType to its int 
        # counterpart. Typical examples are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32
        
        targetType = str(self.samples['train']['targets'].dtype)
        if 'int' in targetType:
            if 'numpy' in repr(dataType):
                if '64' in targetType:
                    targetType = np.int64
                elif '32' in targetType:
                    targetType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in targetType:
                    targetType = torch.int64
                elif '32' in targetType:
                    targetType = torch.int32
        else: # If there is no int, just stick with the given dataType
            targetType = dataType
        
        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        for key in self.samples.keys():
            self.samples[key]['signals'] = changeDataType(
                                                   self.samples[key]['signals'],
                                                   dataType)
            self.samples[key]['targets'] = changeDataType(
                                                   self.samples[key]['targets'],
                                                   targetType)

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if 'torch' in repr(self.dataType):
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device
                
class _dataForClassification(_data):
    # Internal supraclass from which data classes inherit when they are used
    # for classification. This renders the .evaluate() method the same in all
    # cases (how many examples are incorrectly labeled) so justifies the use of
    # another internal class.
    
    def __init__(self):
        
        super().__init__()
    

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim = 1)
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            errorRate = totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            errorRate = totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return errorRate
    
class SourceLocalization_slog(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label
                
    .expandDims(): Adds the feature dimension to the graph signals (i.e. for
        graph signals of shape nSamples x nNodes, turns them into shape
        nSamples x 1 x nNodes, so that they can be handled by general graph
        signal processing techniques that take into account a feature dimension
        by default)

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    errorRate = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): unnormalized probability of each label (shape:
                nDataPoints x nClasses)
            y (dtype.array): correct labels (1-D binary vector, shape:
                nDataPoints)
            tol (float, default = 1e-9): numerical tolerance to consider two
                numbers to be equal
        Output:
            errorRate (float): proportion of incorrect labels

    """

    def __init__(self, G, nTrain, nValid, nTest, sourceNodes, communityList, V, alpha = 1.0, selectMode = 'default',signalMode = 'default', tMax = None,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        N_C = len(sourceNodes[0])
        self.N_C = N_C        
        C = len(sourceNodes)
        print('Number of input nodes per classes:',N_C,'; number of classes',C)
        self.C = C
        # If no tMax is specified, set it the maximum possible.
        if tMax == None:
            tMax = G.N
            
        #\\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order = 'totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
#         sampledSources = np.random.choice(sourceNodes, size = nTotal)
#         # sample diffusion times
#         sampledTimes = np.random.choice(tMax, size = nTotal)
#         # Since the signals are generated as W^t * delta, this reduces to the
#         # selection of a column of W^t (the column corresponding to the source
#         # node). Therefore, we generate an array of size tMax x N x N with all
#         # the powers of the matrix, and then we just simply select the
#         # corresponding column for the corresponding time
#         lastWt = np.eye(G.N, G.N)
#         Wt = lastWt.reshape([1, G.N, G.N])
#         for t in range(1,tMax):
#             lastWt = lastWt @ Wnorm
#             Wt = np.concatenate((Wt, lastWt.reshape([1, G.N, G.N])), axis = 0)
#         x = Wt[sampledTimes, :, sampledSources]
        
        
        # Let's generate slog input x
        sourceNode_set = np.array(sourceNodes)
        communityNode_set = np.array(communityList)
#         communityNode_set = communityList
        sampledSources_index = np.random.choice(np.arange(len(sourceNodes)), size = nTotal)
        sampledSources = sourceNode_set[sampledSources_index]
        sampledCommunities  = communityNode_set[sampledSources_index]      
        x_slog = torch.zeros(nTotal,G.N)
        X0 = torch.zeros(G.N,nTotal)
        N = G.N
        gT = np.ones([N,nTotal])+alpha*np.random.normal(0, 1, [N,nTotal]) 
        if selectMode == 'default':
            for n_t in range(nTotal):
                x0 = torch.zeros(G.N)
                for n_c in range(N_C):
                    if signalMode == 'default':
                        x0[sampledSources[n_t,n_c]] = 1
                    else:
                        x0[sampledSources[n_t,n_c]] = np.random.randn()
                X0[:,n_t] = x0
                gt = gT[:,n_t]
                Gt = torch.tensor(np.dot(V,np.dot(np.diag(gt),np.transpose(V))))
                x_slog[n_t,:] = torch.matmul(Gt,x0)
        else:
            for n_t in range(nTotal):
                x0 = torch.zeros(G.N)
                sampledCommunities_t = sampledCommunities[n_t]
                np.random.shuffle(sampledCommunities_t)
                for n_c in range(N_C):
                    if signalMode == 'default':
                        x0[sampledCommunities_t[n_c]] = 1
                    else:
                        x0[sampledCommunities_t[n_c]] = np.random.randn()
                X0[:,n_t] = x0
                gt = gT[:,n_t]
                Gt = torch.tensor(np.dot(V,np.dot(np.diag(gt),np.transpose(V))))
                x_slog[n_t,:] = torch.matmul(Gt,x0)      
        # Now, we have the signals and the labels
#         signals = x # nTotal x N
        signals = x_slog # nTotal x N        
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.
#         nodesToLabels = {}
#         for it in range(len(sourceNodes)):
#             nodesToLabels[sampledSources_index[it]] = it
#         labels = [nodesToLabels[x] for x in sampledSources_index] # nTotal
        labels = sampledSources_index
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['targets'] = np.array(labels[0:nTrain])
        self.samples['train']['X0'] = X0[:,0:nTrain]
        self.samples['train']['gT'] = gT[:,0:nTrain]        
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['targets'] =np.array(labels[nTrain:nTrain+nValid])
        self.samples['valid']['X0'] = X0[:,nTrain:nTrain+nValid]
        self.samples['valid']['gT'] = gT[:,nTrain:nTrain+nValid]        
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['targets'] =np.array(labels[nTrain+nValid:nTotal])
        self.samples['test']['X0'] = X0[:,nTrain+nValid:nTotal]
        self.samples['test']['gT'] = gT[:,nTrain+nValid:nTotal]        
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
### Graph tools
def get_eig_normalized_adj(gso):
    # Input should be numpy tensor
    N,_ = np.shape(gso)
    d = np.sum(gso, axis=1)
    Lp = np.diag(d) - gso
    Lpn = np.zeros((N,N))
    An = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Lpn[i,j] = Lp[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
            An[i,j] = gso[i,j]/np.sqrt(d[i])/np.sqrt(d[j])
    eigenvalues, V = np.linalg.eig(An)  
    return d,Lpn, eigenvalues, V  
    

    
def normalize_gso(gso):
    eigenvalues, _ = np.linalg.eig(gso)
    return gso / np.max(eigenvalues.real)

def generate_V(N):
    V = torch.randn([N,N])
    V[1,:] = V[1,:]/torch.norm(V[1,:])
    for i in range(2,N):
        for j in range(i-1):
            V[i,:] = V[i,:] - torch.dot(V[i,:],V[j,:])*V[j,:]
        V[i,:] = V[i,:]/torch.norm(V[i,:])
    return V

################################################################
############### Loss function ##################################
################################################################
def myLoss(x_pre, x_ture):
    sizes = x_pre.size()
#     return torch.min(torch.norm(x_pre-x_ture)**2,torch.norm(x_pre+x_true)**2)
    return torch.min(((x_pre-x_ture)**2).mean(),((x_pre+x_ture)**2).mean())
    


#####################################################################
############### Objective functions #################################
#####################################################################
def myFunction(rho_0,eta_0,alpha_1,alpha_2,alpha_3,alpha_4,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, Z, C, K, T):
    # define variables
    K = rho_0.shape[0]
    Z = torch.tensor(Z,requires_grad=True)
    N = Z.shape[1]
    P = int(Z.shape[0]/Z.shape[1])
    NP = int(N*P)

    # Initialization
    x = np.zeros([NP,1])
    x = torch.tensor(x)
    u = np.zeros([NP,1])
    u = torch.tensor(u)
    eta = torch.tensor(np.zeros(1))
    II = np.ones([N,N])
    In = torch.tensor(np.ones(N)) # ones(N,1)
    # define function
    for k in range(K):
        # V update
        # Inverse of ZIk = (rho_0*Z^T*Z + eta_0*II)
        ZIk = rho_0[k]*torch.matmul(torch.transpose(Z,0,1),Z) + eta_0[k]*II
        ZIk_inv = torch.linalg.inv(ZIk)
        v_temp = torch.matmul(torch.transpose(Z,0,1),rho_0[k]*x - u) + (eta_0[k]*C-eta)*In
        v = torch.matmul(ZIk_inv, v_temp)
        
        # X update
        for t in range(T):
            x = alpha_1[k,t]*x + alpha_2[k,t]*torch.matmul(Z,v) + alpha_3[k,t]*u + alpha_4[k,t]*torch.sgn(x)
        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.transpose(In),v) + gamma_3[k]*C
    # y = torch.matmul(z.permute(0, 2, 1).reshape([B, N, K]), h)
    return x

def myFunction2(rho_0,eta_0,alpha_1,alpha_2,alpha_3,alpha_4,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K, T):
    # define variables
    K = rho_0.shape[0]
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()
    N = Y.shape[0]
    P = Y.shape[1]
#     y = Y.reshape((N*P, 1), order='F')    
    NP = int(N*P)

    # Initialization
    x = np.zeros(NP)
    x = torch.tensor(x)
    u = np.zeros(NP)
    u = torch.tensor(u)
    eta = torch.tensor(np.zeros(1))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    # define function
    for k in range(K):
        # V update
        # Inverse of ZIk = (rho_0*Z^T*Z + eta_0*II)
        ZIk = rho_0[k]*torch.matmul(torch.transpose(Z,0,1),Z) + eta_0[k]*II
#         ZIk = rho_0[k]*torch.matmul(ZT,Z) + eta_0[k]*II
        ZIk_inv = torch.inverse(ZIk)
#         ZIk_inv = ZIk
        v_temp = torch.matmul(torch.transpose(Z,0,1),rho_0[k]*x - u) + (eta_0[k]*C-eta)*In
#         v_temp = torch.matmul(ZT,rho_0[k]*x - u) + (eta_0[k]*C-eta)*In
        v = torch.matmul(ZIk_inv, v_temp)
#         if k == 0:
#             print([v_temp.size(),v.size(), ZIk_inv.size(), x.size(),u.size()])
#         if k == K-2:
#             print([v_temp.size(),v.size(), ZIk_inv.size(), x.size(),u.size()])            
        # X update
        for t in range(T):
            x = alpha_1[k,t]*x + alpha_2[k,t]*torch.matmul(Z,v) + alpha_3[k,t]*u + alpha_4[k,t]*torch.sign(x)
        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    # y = torch.matmul(z.permute(0, 2, 1).reshape([B, N, K]), h)
    return x,v

def myFunction3(rho_0,eta_0,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables
    K = rho_0.shape[0]
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()
    N = Y.shape[0]
    P = Y.shape[1]
    NP = int(N*P)
    # Initialization
#     x = np.zeros(NP)
#     x = torch.tensor(x)
    x = torch.randn(NP)
    u = np.zeros(NP)
    u = torch.tensor(u)
    eta = torch.tensor(np.zeros(1))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    # define function
    for k in range(K):
        # V update
        ZIk = rho_0[k]*torch.matmul(torch.transpose(Z,0,1),Z) + eta_0[k]*II
        ZIk_inv = torch.pinverse(ZIk)
        v_temp = torch.matmul(torch.transpose(Z,0,1),rho_0[k]*x - u) + (eta_0[k]*C-eta)*In
        v = torch.matmul(ZIk_inv, v_temp)
        # X update
        X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
        x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction4(rho_1,eta_1,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables
    K = rho_1.shape[0]
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()
    N = Y.shape[0]
    P = Y.shape[1]
    NP = int(N*P)
    # Initialization
#     x = np.zeros(NP)
#     x = torch.tensor(x)
    x = torch.randn(NP)
    u = np.zeros(NP)
    u = torch.tensor(u)
    eta = torch.tensor(np.zeros(1))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    # define function
    for k in range(K):
        # V update
        ZIk = torch.matmul(torch.transpose(Z,0,1),Z) + eta_1[k]*II
        ZIk_inv = torch.inverse(ZIk)
        v_temp = torch.matmul(torch.transpose(Z,0,1),x - rho_1[k]*u) + (eta_1[k]*C-rho_1[k]*eta)*In
        v = torch.matmul(ZIk_inv, v_temp)
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])  
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction_slog_1(rho_1,eta_1,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables
    K = rho_1.shape[0]
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()
    N = Y.shape[0]
    P = Y.shape[1]
    NP = int(N*P)
    # Initialization
#     x = np.zeros(NP)
#     x = torch.tensor(x)
    x = torch.randn(NP)
    u = np.zeros(NP)
    u = torch.tensor(u)
    eta = torch.tensor(np.zeros(1))
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)
    # define function
    for k in range(K):
        # V update
#         ZIk = torch.matmul(torch.transpose(Z,0,1),Z) + eta_1[k]*II
#         ZIk_inv = torch.inverse(ZIk)
        ZIk_inv = fast_inverse(torch.matmul(torch.transpose(Z,0,1),Z),eta_1[k])
        v_temp = torch.matmul(torch.transpose(Z,0,1),x - rho_1[k]*u) + (eta_1[k]*C-rho_1[k]*eta)*In
        v = torch.matmul(ZIk_inv, v_temp)
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])  
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction_ladmm_net_v1(rho_1,rho_2,rho_3,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3, V,Y, C, K):
    # define variables
    C = torch.tensor(C,dtype=torch.double,requires_grad=False)
    C = C.reshape(1)    
    K = rho_1.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.tensor(np.ones([N,N]))
    In = torch.tensor(np.ones(N)) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
#     print([Z.shape])
    Zc = torch.cat([Z,torch.tensor(np.ones([1,N]))],dim=0)
    ZT = Z.t()  
    ZcT = Zc.t()
    # Initialization
    x = torch.randn(NP,dtype=torch.double)
    v = torch.randn(N,dtype=torch.double)
    xc = torch.cat([x,C],dim=0)
    uc = np.zeros(NP+1)
    uc = torch.tensor(uc)

    # define function
    for k in range(K):
        # V update
#         ZIk = torch.matmul(torch.transpose(Z,0,1),Z) + eta_1[k]*II
#         ZIk_inv = torch.inverse(ZIk)
#         v_temp = torch.matmul(torch.transpose(Z,0,1),x - rho_1[k]*u) + (eta_1[k]*C-rho_1[k]*eta)*In
#         v = torch.matmul(ZIk_inv, v_temp)
        I_ZTZ = torch.eye(N) - rho_1[k]*torch.matmul(torch.transpose(Zc,0,1),Zc)
        v_temp = torch.matmul(I_ZTZ, v) + rho_2[k]*torch.matmul(torch.transpose(Zc,0,1),xc) - rho_3[k]*torch.matmul(torch.transpose(Zc,0,1),uc)
        # X update
        X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
        u,C = torch.split(uc,[NP,1])
        x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        xc = torch.cat([x,C],dim=0)
        # M update
        uc = beta_1[k]*uc + beta_2[k]*torch.matmul(Zc,v) + beta_3[k]*xc
    x,C = torch.split(xc,[NP,1])
    return x,v

def myFunction_ladmm_net_v2(rho_1,rho_2,rho_3,rho_4,rho_5,rho_6,rho_7, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables  
    K = rho_1.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = torch.randn(NP) #,dtype=torch.double)
    v = torch.ones(N) #,dtype=torch.double)
    u = torch.tensor(np.zeros(NP))
    eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update
        I_ZTZ = torch.eye(N) - rho_1[k]*torch.matmul(torch.transpose(Z,0,1),Z)
        v_temp = torch.matmul(I_ZTZ - rho_4[k]*II, v) + torch.matmul(torch.transpose(Z,0,1),rho_2[k]*x - rho_3[k]*u)
        v = v_temp + (rho_5[k]*C- rho_6[k]*eta)*In - rho_7[k]*v       
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C

    return x,v

def myFunction_ladmm_net_v3(Phi_1,Phi_2,Phi_3,Phi_4,eta_0, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables 
    K = eta_0.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = torch.randn(NP) #,dtype=torch.double)
    v = torch.ones(N) #,dtype=torch.double)
    u = torch.tensor(np.zeros(NP))
    eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update 
        v = torch.matmul(Phi_1[:,:,k].reshape(N,NP),torch.matmul(Z,v)) + torch.matmul(Phi_2[:,:,k].reshape(N,NP),x) + torch.matmul(Phi_3[:,:,k].reshape(N,NP),u)
        v = v + torch.matmul(Phi_4[:,:,k].reshape(N,N),v) + eta_0[k]*C
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction_ladmm_net_v4(Phi_1,Phi_2,Phi_3,eta_0,eta_1, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables 
    K = eta_0.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = torch.randn(NP) #,dtype=torch.double)
    v = torch.ones(N) #,dtype=torch.double)
    u = torch.tensor(np.zeros(NP))
    eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update 
        v = torch.matmul(Phi_1[:,:,k].reshape(N,N),v) + torch.matmul(Phi_2[:,:,k].reshape(N,N),torch.matmul(torch.transpose(Z,0,1),x)) + torch.matmul(Phi_3[:,:,k].reshape(N,N),torch.matmul(torch.transpose(Z,0,1),u))
        v = v + eta_0[k]*C - eta_1[k]*eta
        v = N*torch.div(v, torch.sum(v))
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
#         uc = beta_1[k]*uc + beta_2[k]*torch.matmul(Zc,v) + beta_3[k]*xc
    return x,v

def myFunction_ladmm_net_v5(Phi_1,Phi_2,Phi_3, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables  
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = torch.randn(NP) #,dtype=torch.double)
    v = torch.ones(N) #,dtype=torch.double)
    u = torch.tensor(np.zeros(NP))

    # define function
    for k in range(K):
        # V update 
        v = torch.matmul(Phi_1[:,:,k].reshape(N,N),v) + torch.matmul(Phi_2[:,:,k].reshape(N,N),torch.matmul(torch.transpose(Z,0,1),x)) + torch.matmul(Phi_3[:,:,k].reshape(N,N),torch.matmul(torch.transpose(Z,0,1),u))
        v = N*torch.div(v, torch.sum(v))
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
    return x,v

def myFunction_ladmm_net_v6(rho_1,rho_2,rho_3,rho_4,rho_0, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # define variables  
    K = rho_1.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = torch.randn(NP) #,dtype=torch.double)
    v = torch.ones(N) #,dtype=torch.double)
    u = torch.tensor(np.zeros(NP))
    eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update
        I_ZTZ = rho_0[k]*torch.eye(N) - rho_1[k]*torch.matmul(torch.transpose(Z,0,1),Z)
        v_temp = torch.matmul(I_ZTZ - rho_3[k]*II, v) + torch.matmul(torch.transpose(Z,0,1),rho_2[k]*x - rho_0[k]* u)
        v = v_temp + (rho_4[k]*C- rho_0[k]*eta)*In    
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction_ladmm_net_v7(rho_1,rho_2,rho_3,rho_4,rho_0, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K,L):
    # define variables  
    # 
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = torch.randn(NP) #,dtype=torch.double)
#     v = torch.ones(N) #,dtype=torch.double)
    v = torch.randn(N)
    u = torch.tensor(np.zeros(NP))
    eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update
        for l in range(L):
            I_ZTZ = rho_0[k,l]*torch.eye(N) - rho_1[k,l]*torch.matmul(torch.transpose(Z,0,1),Z)
            v_temp = torch.matmul(I_ZTZ - rho_3[k,l]*II, v) + torch.matmul(torch.transpose(Z,0,1),rho_2[k,l]*x - rho_0[k,l]* u)
            v = v_temp + (rho_4[k,l]*C- rho_0[k,l]*eta)*In
            v = N*torch.div(v, torch.sum(v)+1e-10)
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction_ladmm_net_v8(rho_1,rho_2,rho_3,rho_4,rho_0, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K,L):
    # No C, eta constraint
    # define variables  
#     K = rho_1.shape[0]
#     L = rho_1.shape[1]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = torch.randn(NP) #,dtype=torch.double)
#     v = torch.ones(N) #,dtype=torch.double)
    v = torch.randn(N)
    u = torch.tensor(np.zeros(NP))

    # define function
    for k in range(K):
        # V update
        for l in range(L):
            I_ZTZ = rho_0[k,l]*torch.eye(N) - rho_1[k,l]*torch.matmul(torch.transpose(Z,0,1),Z)
            v_temp = torch.matmul(I_ZTZ, v) + torch.matmul(torch.transpose(Z,0,1),rho_2[k,l]*x - rho_0[k,l]* u)
            v = v_temp - rho_3[k,l]*v
            v = N*torch.div(v, torch.sum(v)+1e-10)
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
    return x,v

def myFunction_ladmm_net_v9(rho_0,rho_1,rho_2,rho_3,rho_4,rho_5,rho_6,rho_7, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # Make everything linear
    # define variables  
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = Variable(torch.randn(NP), requires_grad=True)
#     x = torch.randn(NP) #,dtype=torch.double)
    v = Variable(torch.randn(N), requires_grad=True)
#     v = torch.ones(N) #,dtype=torch.double)
#     u = torch.tensor(np.zeros(NP))
    u = Variable(torch.randn(NP), requires_grad=True)
    eta = Variable(torch.randn(1), requires_grad=True)
#     lmbd_var = Variable(torch.randn(K), requires_grad=True)
#     eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update
        I_ZTZ = rho_0[k]*torch.eye(N) - rho_1[k]*torch.matmul(torch.transpose(Z,0,1),Z)
        v_temp = torch.matmul(I_ZTZ - rho_2[k]*II, v) + torch.matmul(torch.transpose(Z,0,1),rho_3[k]*x - rho_4[k]*u)
        v = v_temp + (rho_5[k]*C- rho_6[k]*eta)*In - rho_7[k]*v       
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
#     lmbd = lmbd_var
    return x,v

def myFunction_ladmm_net_v9_1(rho_1,rho_2,rho_3,rho_4,rho_5,rho_6,rho_7, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # Make everything linear
    # Remove rho_0 from v9
    # define variables  
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = Variable(torch.randn(NP), requires_grad=True)
#     x = torch.randn(NP) #,dtype=torch.double)
    v = Variable(torch.randn(N), requires_grad=True)
#     v = torch.ones(N) #,dtype=torch.double)
#     u = torch.tensor(np.zeros(NP))
    u = Variable(torch.randn(NP), requires_grad=True)
    eta = Variable(torch.randn(1), requires_grad=True)
#     lmbd_var = Variable(torch.randn(K), requires_grad=True)
#     eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update
        I_ZTZ = torch.eye(N) - rho_1[k]*torch.matmul(torch.transpose(Z,0,1),Z)
        v_temp = torch.matmul(I_ZTZ - rho_2[k]*II, v) + torch.matmul(torch.transpose(Z,0,1),rho_3[k]*x - rho_4[k]*u)
        v = v_temp + (rho_5[k]*C- rho_6[k]*eta)*In - rho_7[k]*v       
        # X update
#         X_update = torch.nn.Softshrink(lambd = lmbd[k].item())
#         x = X_update(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u)
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
#     lmbd = lmbd_var
    return x,v

def myFunction_ladmm_net_v9_lstm(lstm, hidden_size, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # Make everything linear 
    # Remove rho_0 from v9
    # Add LSTM to compute the step size
    # define variables  
    num_layers = 1 # number of layers of lstm
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = Variable(torch.randn(NP), requires_grad=True)
    v = Variable(torch.randn(N), requires_grad=True)
    u = Variable(torch.randn(NP), requires_grad=True)
    eta = Variable(torch.randn(1), requires_grad=True)

    # define function
    for k in range(K):
        # V update
        input = torch.cat((torch.matmul(torch.transpose(Z,0,1),torch.matmul(Z,v)-x),torch.matmul(torch.transpose(Z,0,1),u), In*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C), In*eta),0)
        input = torch.reshape(input, (1,1, input.size(0)))
        h0 = Variable(torch.zeros(num_layers, input.size(0), hidden_size)) #hidden state
        c0 = Variable(torch.zeros(num_layers, input.size(0), hidden_size)) #internal state  

        output, (hn, cn) = lstm(input, (h0, c0))  
        step = hn.view(v.size())
#         v = v - step[0,0,0]*torch.matmul(torch.transpose(Z,0,1),torch.matmul(Z,v)-x) - step[0,0,1]*torch.matmul(torch.transpose(Z,0,1),u) - step[0,0,2]*In*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C) - step[0,0,3]*eta   
#         print(v.size(),step.size())
        v = v - step

        # X update
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction_ladmm_net_v9_lstm_2(lstm, num_layers,hidden_size,phi, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # Make everything linear 
    # Remove rho_0 from v9
    # Add LSTM to compute the step size
    # define variables  
#     num_layers = 1 # number of layers of lstm
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = Variable(torch.randn(NP), requires_grad=True)
    v = Variable(torch.randn(N), requires_grad=True)
    u = Variable(torch.randn(NP), requires_grad=True)
    eta = Variable(torch.randn(1), requires_grad=True)
    h0 = Variable(torch.zeros(num_layers, 1, hidden_size)) #hidden state
    c0 = Variable(torch.zeros(num_layers, 1, hidden_size)) #internal state 
        
    # define function
    for k in range(K):
        # V update
        input = torch.cat((torch.matmul(torch.transpose(Z,0,1),torch.matmul(Z,v)-x),torch.matmul(torch.transpose(Z,0,1),u), In*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C), eta),0)
        input = torch.reshape(input, (1,1, input.size(0)))
#         h0 = Variable(torch.zeros(num_layers, input.size(0), hidden_size)) #hidden state
#         c0 = Variable(torch.zeros(num_layers, input.size(0), hidden_size)) #internal state  

        output, (hn, cn) = lstm(input, (h0, c0))  
        h0 = hn
        c0 = cn
        step = torch.matmul(phi,hn.view(num_layers,hidden_size))
#         print(step.size())
        step = step.view(v.size())
#         step = step[:,0].view(v.size())
#         v = v - step[0,0,0]*torch.matmul(torch.transpose(Z,0,1),torch.matmul(Z,v)-x) - step[0,0,1]*torch.matmul(torch.transpose(Z,0,1),u) - step[0,0,2]*In*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C) - step[0,0,3]*eta   
#         print(v.size(),step.size())
        v = v - step

        # X update
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v

def myFunction_ladmm_net_v9_lstm_3(lstm,phi,num_layers, hidden_size,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # Make everything linear 
    # Remove rho_0 from v9
    # Add LSTM to compute the step size
    # define variables  
#     num_layers = 1 # number of layers of lstm
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = Variable(torch.randn(NP), requires_grad=True)
    v = Variable(torch.randn(N), requires_grad=True)
    u = Variable(torch.randn(NP), requires_grad=True)
    eta = Variable(torch.randn(1), requires_grad=True)
    h0 = Variable(torch.zeros(num_layers, 1, hidden_size)) #hidden state
    c0 = Variable(torch.zeros(num_layers, 1, hidden_size)) #internal state   
    # define function
    for k in range(K):
        # V update
        input = torch.matmul(torch.transpose(Z,0,1),1*(torch.matmul(Z,v)-x)+u) + In*(1*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C) + eta)
        input = torch.reshape(input, (1,1, input.size(0)))
        output, (hn, cn) = lstm(input, (h0, c0))  
        h0 = hn
        c0 = cn
        step = torch.matmul(phi,hn.view(num_layers,hidden_size))
#         print(step.size())
#         step = hn.view(v.size())
        step = step[0,:].view(v.size())
#         print(v.size(),step.size())
        v = v - step

        # X update
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v



def myFunction_ladmm_net_v9_lstm_4(lstm,phi,b,num_layers, hidden_size,lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K):
    # Make everything linear 
    # Remove rho_0 from v9
    # Add LSTM to compute the step size
    # define variables  
#     num_layers = 1 # number of layers of lstm
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = Variable(torch.randn(NP), requires_grad=True)
    v = Variable(torch.randn(N), requires_grad=True)
    u = Variable(torch.randn(NP), requires_grad=True)
    eta = Variable(torch.randn(1), requires_grad=True)
    h0 = Variable(torch.zeros(num_layers, 1, hidden_size)) #hidden state
    c0 = Variable(torch.zeros(num_layers, 1, hidden_size)) #internal state   
    # define function
    for k in range(K):
        # V update
        input = torch.matmul(torch.transpose(Z,0,1),1*(torch.matmul(Z,v)-x)+u) + In*(1*(torch.matmul(torch.tensor(np.ones([1,N])),v) - C) + eta)
        input = torch.reshape(input, (1,1, input.size(0)))
        output, (hn, cn) = lstm(input, (h0, c0))  
        h0 = hn
        c0 = cn
        step = torch.matmul(phi,hn.view(num_layers*hidden_size)) + b
#         print(step.size())
#         step = hn.view(v.size())
#         step = step[0,:].view(v.size())
        step = step.view(v.size())    
#         print(v.size(),step.size())
        v = v - step

        # X update
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
    return x,v


def myFunction_ladmm_net_v10(rho_1,rho_2,rho_3,rho_4,rho_5,rho_6,rho_7, lmbd,alpha_1,alpha_2,beta_1,beta_2,beta_3,gamma_1,gamma_2,gamma_3, V,Y, C, K,L):
    # Make everything linear
    # define variables  
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
    II = torch.ones(N,N)
    In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
    ZT = Z.t()  
    # Initialization
    x = Variable(torch.randn(NP), requires_grad=True)
#     x = torch.randn(NP) #,dtype=torch.double)
    v = Variable(torch.randn(N), requires_grad=True)
#     v = torch.ones(N) #,dtype=torch.double)
#     u = torch.tensor(np.zeros(NP))
    u = Variable(torch.randn(NP), requires_grad=True)
    eta = Variable(torch.randn(1), requires_grad=True)
#     lmbd_var = Variable(torch.randn(K), requires_grad=True)
#     eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update
        for l in range(L):
#             I_ZTZ = rho_0[k,l]*torch.eye(N) - rho_1[k,l]*torch.matmul(torch.transpose(Z,0,1),Z)
            I_ZTZ = torch.eye(N) - rho_1[k,l]*torch.matmul(torch.transpose(Z,0,1),Z)            
            v_temp = torch.matmul(I_ZTZ - rho_2[k,l]*II, v) + torch.matmul(torch.transpose(Z,0,1),rho_3[k,l]*x - rho_4[k,l]*u)
            v = v_temp + (rho_5[k,l]*C- rho_6[k,l]*eta)*In - rho_7[k,l]*v       
        # X update

        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = beta_1[k]*u + beta_2[k]*torch.matmul(Z,v) + beta_3[k]*x
        eta = gamma_1[k]*eta + gamma_2[k]*torch.matmul(torch.tensor(np.ones([1,N])),v) + gamma_3[k]*C
#     lmbd = lmbd_var
    return x,v

def myFunction_ladmm_net_v11(W_1,w_2,rho_1,rho_2, lmbd,alpha_1,alpha_2,beta,gamma, V,Y, C, K):
    # Make everything linear
    # define variables  
    K = lmbd.shape[0]
    N = Y.shape[0]
    P = Y.shape[1]
#     II = torch.ones(N,N)
#     In = torch.ones(N) # ones(N,1)    
    NP = int(N*P)
    Z = torch.tensor(linalg.khatri_rao(np.dot(np.transpose(Y),V),V),requires_grad=False)
#     ZT = Z.t()  
    # Initialization
    x = Variable(torch.randn(NP), requires_grad=True)
#     x = torch.randn(NP) #,dtype=torch.double)
    v = Variable(torch.randn(N), requires_grad=True)
#     v = torch.ones(N) #,dtype=torch.double)
#     u = torch.tensor(np.zeros(NP))
    u = Variable(torch.randn(NP), requires_grad=True)
    eta = Variable(torch.randn(1), requires_grad=True)
#     lmbd_var = Variable(torch.randn(K), requires_grad=True)
#     eta = torch.tensor(np.zeros(1))

    # define function
    for k in range(K):
        # V update
#         print(W_1[:,:,k].size(), rho_1[:,k].size())
        v = v - torch.matmul(W_1[:,:,k].view(N,NP), u + rho_1[:,k].view(NP)*(torch.matmul(Z,v) - x)) - w_2[:,k].view(N)*(rho_2[k]*(torch.sum(v) - C) + eta)    
        # X update
        x = softshrink(alpha_1[k]*torch.matmul(Z,v) + alpha_2[k]*u,lmbd[k])        
        # M update
        u = u + beta[:,k].view(NP)*(torch.matmul(Z,v) - x)
        eta = eta + gamma[k]*(torch.matmul(torch.tensor(np.ones([1,N])),v)- C)
    return x,v

######################################################################################################################
class GraphLADMMnet11(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters    
        self.rho_1 = nn.Parameter(torch.randn(self.N*self.P,self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.W_1 = nn.Parameter(torch.randn(self.N,self.N*self.P,self.K))
        self.w_2 = nn.Parameter(torch.randn(self.N,self.K))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta = nn.Parameter(torch.randn(self.N*self.P,self.K))       
        self.gamma = nn.Parameter(torch.randn(self.K))  
  
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_1 = nn.Parameter(torch.randn(self.N*self.P,self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.W_1 = nn.Parameter(torch.randn(self.N,self.N*self.P,self.K))
        self.w_2 = nn.Parameter(torch.randn(self.N,self.K))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta = nn.Parameter(torch.randn(self.N*self.P,self.K))       
        self.gamma = nn.Parameter(torch.randn(self.K))       

    def forward(self, Y):
        return myFunction_ladmm_net_v11(self.W_1,self.w_2,self.rho_1,self.rho_2,self.lmbd,self.alpha_1,self.alpha_2,self.beta,self.gamma, self.V,Y, self.C, self.K)
    
class GraphLADMMnet10(nn.Module):
    def __init__(self, V, N, P, C, K,L, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.L = int(L)
        self.gso = gso
        self.V = V
        # Parameters
#         self.rho_0 = nn.Parameter(torch.randn(self.K,self.L))        
        self.rho_1 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_2 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_3 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_4 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_5 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_6 = nn.Parameter(torch.randn(self.K,self.L))      
        self.rho_7 = nn.Parameter(torch.randn(self.K,self.L))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
#         self.rho_0 = nn.Parameter(torch.randn(self.K,self.L))        
        self.rho_1 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_2 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_3 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_4 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_5 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_6 = nn.Parameter(torch.randn(self.K,self.L))      
        self.rho_7 = nn.Parameter(torch.randn(self.K,self.L))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v10(self.rho_1,self.rho_2,self.rho_3,self.rho_4,self.rho_5,self.rho_6,self.rho_7,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K,self.L)

    
class GraphLADMMnet9_lstm_4(nn.Module):
    def __init__(self, hidden_size, num_layers, V, N, P, C, K, gso):
        super(GraphLADMMnet9_lstm_4, self).__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        input_size = N
#         self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
#         self.seq_length = seq_length #sequence length
#         self.h0 = 
#         self.c0 = 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm        
        # Parameters
#         self.rho_0 = nn.Parameter(torch.randn(self.K)) 
#         self.rho_1 = nn.Parameter(torch.randn(self.K))
#         self.rho_2 = nn.Parameter(torch.randn(self.K))
#         self.rho_3 = nn.Parameter(torch.randn(self.K))
#         self.rho_4 = nn.Parameter(torch.randn(self.K))
#         self.rho_5 = nn.Parameter(torch.randn(self.K))
#         self.rho_6 = nn.Parameter(torch.randn(self.K))      
#         self.rho_7 = nn.Parameter(torch.randn(self.K))  
        self.b = nn.Parameter(torch.randn(self.N))
        self.phi = nn.Parameter(torch.randn(self.N, self.num_layers*self.hidden_size))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
#         self.rho_0 = nn.Parameter(torch.randn(self.K))        
#         self.rho_1 = nn.Parameter(torch.randn(self.K))
#         self.rho_2 = nn.Parameter(torch.randn(self.K))
#         self.rho_3 = nn.Parameter(torch.randn(self.K))
#         self.rho_4 = nn.Parameter(torch.randn(self.K))
#         self.rho_5 = nn.Parameter(torch.randn(self.K))
#         self.rho_6 = nn.Parameter(torch.randn(self.K))  
#         self.rho_7 = nn.Parameter(torch.randn(self.K))     
        self.b = nn.Parameter(torch.randn(self.N))
        self.phi = nn.Parameter(torch.randn(self.N, self.num_layers*self.hidden_size))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):    
        return myFunction_ladmm_net_v9_lstm_4(self.lstm,self.phi,self.b, self.num_layers,self.hidden_size,  self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)
            
        
class GraphLADMMnet9_lstm_3(nn.Module):
    def __init__(self, hidden_size, num_layers, V, N, P, C, K, gso):
        super(GraphLADMMnet9_lstm_3, self).__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        input_size = N
#         self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
#         self.seq_length = seq_length #sequence length
#         self.h0 = 
#         self.c0 = 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm        
        # Parameters
#         self.rho_0 = nn.Parameter(torch.randn(self.K)) 
#         self.rho_1 = nn.Parameter(torch.randn(self.K))
#         self.rho_2 = nn.Parameter(torch.randn(self.K))
#         self.rho_3 = nn.Parameter(torch.randn(self.K))
#         self.rho_4 = nn.Parameter(torch.randn(self.K))
#         self.rho_5 = nn.Parameter(torch.randn(self.K))
#         self.rho_6 = nn.Parameter(torch.randn(self.K))      
#         self.rho_7 = nn.Parameter(torch.randn(self.K))  
        self.phi = nn.Parameter(torch.randn(1,self.num_layers))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
#         self.rho_0 = nn.Parameter(torch.randn(self.K))        
#         self.rho_1 = nn.Parameter(torch.randn(self.K))
#         self.rho_2 = nn.Parameter(torch.randn(self.K))
#         self.rho_3 = nn.Parameter(torch.randn(self.K))
#         self.rho_4 = nn.Parameter(torch.randn(self.K))
#         self.rho_5 = nn.Parameter(torch.randn(self.K))
#         self.rho_6 = nn.Parameter(torch.randn(self.K))  
#         self.rho_7 = nn.Parameter(torch.randn(self.K))     
        self.phi = nn.Parameter(torch.randn(1,self.num_layers))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):    
        return myFunction_ladmm_net_v9_lstm_3(self.lstm,self.phi, self.num_layers,self.hidden_size,  self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)
            
class GraphLADMMnet9_lstm_2(nn.Module):
    def __init__(self, hidden_size, num_layers, V, N, P, C, K, gso):
        super(GraphLADMMnet9_lstm_2, self).__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        input_size = 3*N+1
#         self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
#         self.seq_length = seq_length #sequence length
#         self.h0 = 
#         self.c0 = 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm        
        # Parameters
#         self.rho_0 = nn.Parameter(torch.randn(self.K)) 
#         self.rho_1 = nn.Parameter(torch.randn(self.K))
#         self.rho_2 = nn.Parameter(torch.randn(self.K))
#         self.rho_3 = nn.Parameter(torch.randn(self.K))
#         self.rho_4 = nn.Parameter(torch.randn(self.K))
#         self.rho_5 = nn.Parameter(torch.randn(self.K))
#         self.rho_6 = nn.Parameter(torch.randn(self.K))      
#         self.rho_7 = nn.Parameter(torch.randn(self.K))  
        self.phi = nn.Parameter(torch.randn(1,self.num_layers))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
#         self.rho_0 = nn.Parameter(torch.randn(self.K))        
#         self.rho_1 = nn.Parameter(torch.randn(self.K))
#         self.rho_2 = nn.Parameter(torch.randn(self.K))
#         self.rho_3 = nn.Parameter(torch.randn(self.K))
#         self.rho_4 = nn.Parameter(torch.randn(self.K))
#         self.rho_5 = nn.Parameter(torch.randn(self.K))
#         self.rho_6 = nn.Parameter(torch.randn(self.K))  
#         self.rho_7 = nn.Parameter(torch.randn(self.K))     
        self.phi = nn.Parameter(torch.randn(1,self.num_layers))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):    
        return myFunction_ladmm_net_v9_lstm_2(self.lstm, self.num_layers,self.hidden_size, self.phi, self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)
    
class GraphLADMMnet9_lstm(nn.Module):
    def __init__(self, hidden_size, num_layers, V, N, P, C, K, gso):
        super(GraphLADMMnet9_lstm, self).__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        input_size = 4*N
#         self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
#         self.seq_length = seq_length #sequence length
#         self.h0 = 
#         self.c0 = 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm        
        # Parameters
#         self.rho_0 = nn.Parameter(torch.randn(self.K)) 
#         self.rho_1 = nn.Parameter(torch.randn(self.K))
#         self.rho_2 = nn.Parameter(torch.randn(self.K))
#         self.rho_3 = nn.Parameter(torch.randn(self.K))
#         self.rho_4 = nn.Parameter(torch.randn(self.K))
#         self.rho_5 = nn.Parameter(torch.randn(self.K))
#         self.rho_6 = nn.Parameter(torch.randn(self.K))      
#         self.rho_7 = nn.Parameter(torch.randn(self.K))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
#         self.rho_0 = nn.Parameter(torch.randn(self.K))        
#         self.rho_1 = nn.Parameter(torch.randn(self.K))
#         self.rho_2 = nn.Parameter(torch.randn(self.K))
#         self.rho_3 = nn.Parameter(torch.randn(self.K))
#         self.rho_4 = nn.Parameter(torch.randn(self.K))
#         self.rho_5 = nn.Parameter(torch.randn(self.K))
#         self.rho_6 = nn.Parameter(torch.randn(self.K))  
#         self.rho_7 = nn.Parameter(torch.randn(self.K))         
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):    
        return myFunction_ladmm_net_v9_lstm(self.lstm, self.hidden_size, self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)
    
class GraphLADMMnet9_1(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters
#         self.rho_0 = nn.Parameter(torch.randn(self.K))        
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))
        self.rho_4 = nn.Parameter(torch.randn(self.K))
        self.rho_5 = nn.Parameter(torch.randn(self.K))
        self.rho_6 = nn.Parameter(torch.randn(self.K))      
        self.rho_7 = nn.Parameter(torch.randn(self.K))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
#         self.rho_0 = nn.Parameter(torch.randn(self.K))        
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))
        self.rho_4 = nn.Parameter(torch.randn(self.K))
        self.rho_5 = nn.Parameter(torch.randn(self.K))
        self.rho_6 = nn.Parameter(torch.randn(self.K))  
        self.rho_7 = nn.Parameter(torch.randn(self.K))         
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v9_1(self.rho_1,self.rho_2,self.rho_3,self.rho_4,self.rho_5,self.rho_6,self.rho_7,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)
    
class GraphLADMMnet9(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_0 = nn.Parameter(torch.randn(self.K))        
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))
        self.rho_4 = nn.Parameter(torch.randn(self.K))
        self.rho_5 = nn.Parameter(torch.randn(self.K))
        self.rho_6 = nn.Parameter(torch.randn(self.K))      
        self.rho_7 = nn.Parameter(torch.randn(self.K))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_0 = nn.Parameter(torch.randn(self.K))        
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))
        self.rho_4 = nn.Parameter(torch.randn(self.K))
        self.rho_5 = nn.Parameter(torch.randn(self.K))
        self.rho_6 = nn.Parameter(torch.randn(self.K))  
        self.rho_7 = nn.Parameter(torch.randn(self.K))         
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v9(self.rho_0,self.rho_1,self.rho_2,self.rho_3,self.rho_4,self.rho_5,self.rho_6,self.rho_7,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)

class GraphLADMMnet8(nn.Module):
    def __init__(self, V, N, P, C, K,L, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.L = int(L)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_1 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_0 = nn.Parameter(torch.randn(self.K,self.L))        
        self.rho_2 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_3 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_4 = nn.Parameter(torch.randn(self.K,self.L))       
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_1 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_0 = nn.Parameter(torch.randn(self.K,self.L))        
        self.rho_2 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_3 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_4 = nn.Parameter(torch.randn(self.K,self.L))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v8(self.rho_1,self.rho_2,self.rho_3,self.rho_4,self.rho_0,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K,self.L)
    
class GraphLADMMnet7(nn.Module):
    def __init__(self, V, N, P, C, K,L, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.L = int(L)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_1 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_0 = nn.Parameter(torch.randn(self.K,self.L))        
        self.rho_2 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_3 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_4 = nn.Parameter(torch.randn(self.K,self.L))       
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_1 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_0 = nn.Parameter(torch.randn(self.K,self.L))        
        self.rho_2 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_3 = nn.Parameter(torch.randn(self.K,self.L))
        self.rho_4 = nn.Parameter(torch.randn(self.K,self.L))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v7(self.rho_1,self.rho_2,self.rho_3,self.rho_4,self.rho_0,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K,self.L)

class GraphLADMMnet6(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_0 = nn.Parameter(torch.randn(self.K))        
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))
        self.rho_4 = nn.Parameter(torch.randn(self.K))       
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_0 = nn.Parameter(torch.randn(self.K))         
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))
        self.rho_4 = nn.Parameter(torch.randn(self.K))       
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v6(self.rho_1,self.rho_2,self.rho_3,self.rho_4,self.rho_0,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)


class GraphLADMMnet5(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.NP = int(N*P)
        self.gso = gso
        self.V = V
        # Parameters
        self.Phi_1 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.Phi_2 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.Phi_3 = nn.Parameter(torch.randn(self.N,self.N,self.K))      
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.Phi_1 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.Phi_2 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.Phi_3 = nn.Parameter(torch.randn(self.N,self.N,self.K))            
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v5(self.Phi_1,self.Phi_2,self.Phi_3,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)     
    
class GraphLADMMnet4(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.NP = int(N*P)
        self.gso = gso
        self.V = V
        # Parameters
        self.Phi_1 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.Phi_2 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.Phi_3 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))    
        self.eta_1 = nn.Parameter(torch.randn(self.K))         
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.Phi_1 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.Phi_2 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.Phi_3 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.eta_1 = nn.Parameter(torch.randn(self.K))         
        self.eta_0 = nn.Parameter(torch.randn(self.K))              
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v4(self.Phi_1,self.Phi_2,self.Phi_3,self.eta_0,self.eta_1,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)                                                                                                                                                     
class GraphLADMMnet3(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.NP = int(N*P)
        self.gso = gso
        self.V = V
        # Parameters
        self.Phi_1 = nn.Parameter(torch.randn(self.N,self.NP,self.K))
        self.Phi_2 = nn.Parameter(torch.randn(self.N,self.NP,self.K))
        self.Phi_3 = nn.Parameter(torch.randn(self.N,self.NP,self.K))
        self.Phi_4 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))      
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.Phi_1 = nn.Parameter(torch.randn(self.N,self.NP,self.K))
        self.Phi_2 = nn.Parameter(torch.randn(self.N,self.NP,self.K))
        self.Phi_3 = nn.Parameter(torch.randn(self.N,self.NP,self.K))
        self.Phi_4 = nn.Parameter(torch.randn(self.N,self.N,self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))              
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v3(self.Phi_1,self.Phi_2,self.Phi_3,self.Phi_4,self.eta_0,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)
    
class GraphLADMMnet2(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))
        self.rho_4 = nn.Parameter(torch.randn(self.K))
        self.rho_5 = nn.Parameter(torch.randn(self.K))
        self.rho_6 = nn.Parameter(torch.randn(self.K))      
        self.rho_7 = nn.Parameter(torch.randn(self.K))        
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))   
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))
        self.rho_4 = nn.Parameter(torch.randn(self.K))
        self.rho_5 = nn.Parameter(torch.randn(self.K))
        self.rho_6 = nn.Parameter(torch.randn(self.K))  
        self.rho_7 = nn.Parameter(torch.randn(self.K))         
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         

    def forward(self, Y):
        return myFunction_ladmm_net_v2(self.rho_1,self.rho_2,self.rho_3,self.rho_4,self.rho_5,self.rho_6,self.rho_7,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V,Y, self.C, self.K)
    
class GraphLADMMnet1(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))      
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))        
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.rho_2 = nn.Parameter(torch.randn(self.K))
        self.rho_3 = nn.Parameter(torch.randn(self.K))      
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))      
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))        

    def forward(self, Y):
        return myFunction_ladmm_net_v1(self.rho_1,self.rho_2,self.rho_3,self.lmbd,self.alpha_1,self.alpha_2,self.beta_1,self.beta_2,self.beta_3, self.V,Y, self.C, self.K)
    
class GraphSLoG_v1(nn.Module):
    def __init__(self, V, N, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.eta_1 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.eta_1 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  

    def forward(self, Y):
        return myFunction_slog_1(self.rho_1,self.eta_1,self.lmbd,self.alpha_1,self.alpha_2, self.beta_1, self.beta_2, self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V, Y, self.C, self.K)
    
class GraphADMMnet5(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.eta_1 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_1 = nn.Parameter(torch.randn(self.K))
        self.eta_1 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  

    def forward(self, Y):
        return myFunction4(self.rho_1,self.eta_1,self.lmbd,self.alpha_1,self.alpha_2, self.beta_1, self.beta_2, self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V, Y, self.C, self.K)
    
class GraphADMMnet4(nn.Module):
    def __init__(self, V, N, P, C, K, gso):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # gso: graph shift operator
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.gso = gso
        self.V = V
        # Parameters
        self.rho_0 = nn.Parameter(torch.randn(self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_0 = nn.Parameter(torch.randn(self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  

    def forward(self, Y):
        return myFunction3(self.rho_0,self.eta_0,self.lmbd,self.alpha_1,self.alpha_2, self.beta_1, self.beta_2, self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V, Y, self.C, self.K)
    
class GraphADMMnet3(nn.Module):
    def __init__(self, V, N, P, C, K):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.K = int(K)
        self.V = V
        # Parameters
        self.rho_0 = nn.Parameter(torch.randn(self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))
        self.lmbd = nn.Parameter(torch.randn(self.K))
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_0 = nn.Parameter(torch.randn(self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        self.alpha_1 = nn.Parameter(torch.randn(self.K))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K))  

    def forward(self, Y):
        return myFunction3(self.rho_0,self.eta_0,self.lmbd,self.alpha_1,self.alpha_2, self.beta_1, self.beta_2, self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V, Y, self.C, self.K)
    
class GraphADMMnet2(nn.Module):
    def __init__(self, V, N, P, C, K, T):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # T: number of iterations in X^(k) update block
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.T = int(T)
        self.K = int(K)
        self.V = V
        # Parameters
        self.rho_0 = nn.Parameter(torch.randn(self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))
#         self.rho_0 = nn.Parameter(torch.randn(1))
#         self.eta_0 = nn.Parameter(torch.randn(1))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        self.alpha_1 = nn.Parameter(torch.randn(self.K,self.T))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K,self.T))  
        self.alpha_3 = nn.Parameter(torch.randn(self.K,self.T))    
        self.alpha_4 = nn.Parameter(torch.randn(self.K,self.T))         
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_0 = nn.Parameter(torch.randn(self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))
#         self.rho_0 = nn.Parameter(torch.randn(1))
#         self.eta_0 = nn.Parameter(torch.randn(1))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        self.alpha_1 = nn.Parameter(torch.randn(self.K,self.T))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K,self.T))  
        self.alpha_3 = nn.Parameter(torch.randn(self.K,self.T))    
        self.alpha_4 = nn.Parameter(torch.randn(self.K,self.T))  

    def forward(self, Y):
        return myFunction2(self.rho_0,self.eta_0,self.alpha_1,self.alpha_2, self.alpha_3, self.alpha_4, self.beta_1, self.beta_2, self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, self.V, Y, self.C, self.K, self.T)
    
class GraphADMMnet(nn.Module):
    def __init__(self, N, P, C, K, T):
        super().__init__()
        # Z: computes Zv = y
        # N: number of nodes
        # P: number of observations
        # C: constrain constant, ones*v = C
        # K: number of layers
        # T: number of iterations in X^(k) update block
        # self.Z = torch.tensor(Z)
        self.N = int(N)
        self.P = int(P)
        self.C = int(C)
        self.T = int(T)
        self.K = int(K)
        # Parameters
        self.rho_0 = nn.Parameter(torch.randn(self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        self.alpha_1 = nn.Parameter(torch.randn(self.K,self.T))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K,self.T))  
        self.alpha_3 = nn.Parameter(torch.randn(self.K,self.T))    
        self.alpha_4 = nn.Parameter(torch.randn(self.K,self.T))         
        # Initialization
        self.reset_parameters()
        
    def reset_parameters(self):
        self.rho_0 = nn.Parameter(torch.randn(self.K))
        self.eta_0 = nn.Parameter(torch.randn(self.K))
        self.beta_1 = nn.Parameter(torch.randn(self.K))
        self.beta_2 = nn.Parameter(torch.randn(self.K))  
        self.beta_3 = nn.Parameter(torch.randn(self.K))  
        self.gamma_1 = nn.Parameter(torch.randn(self.K))  
        self.gamma_2 = nn.Parameter(torch.randn(self.K)) 
        self.gamma_3 = nn.Parameter(torch.randn(self.K))         
        self.alpha_1 = nn.Parameter(torch.randn(self.K,self.T))      
        self.alpha_2 = nn.Parameter(torch.randn(self.K,self.T))  
        self.alpha_3 = nn.Parameter(torch.randn(self.K,self.T))    
        self.alpha_4 = nn.Parameter(torch.randn(self.K,self.T))  

    def forward(self, Z):
        return myFunction(self.rho_0,self.eta_0,self.alpha_1,self.alpha_2, self.alpha_3, self.alpha_4, self.beta_1, self.beta_2, self.beta_3,self.gamma_1,self.gamma_2,self.gamma_3, Z, self.C, self.K, self.T)
