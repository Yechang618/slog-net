from SLOGmodules import SLOGtools as SLOGtools
from SLOGmodules import SLOGdata as SLOGdata
import numpy as np
from numpy import linalg as LA

def admmExperiment(nNodes, S, P, L, N_realiz, **kwargs):
   
    ## Assertations
    assert N_realiz > 0
    assert nNodes >= S
    assert nNodes >= L
    
    ## Simulation parameters
    # simuParas should include:
    # alpha
    # rho_0
    # eta_0
    # C
    # N_ite
    # max_re    
    if 'simuParas' in kwargs.keys():
        simuParas = kwargs['simuParas']
    else:
        simuParas = {}
    
    if 'alpha' in simuParas.keys():
        alpha = simuParas['alpha']
    else:
        alpha = 1.0
        simuParas['alpha'] = alpha

    if 'rho_0' in simuParas.keys():
        rho_0 = simuParas['rho_0']
    else:
        rho_0 = 1.0
        simuParas['rho_0'] = rho_0

    if 'eta_0' in simuParas.keys():
        eta_0 = simuParas['eta_0']
    else:
        eta_0 = 1.0
        simuParas['eta_0'] = eta_0
        
    if 'C' in simuParas.keys():
        C = simuParas['C']
    else:
        C = nNodes
        simuParas['C'] = C
        
    if 'N_ite' in simuParas.keys():
        N_ite = simuParas['N_ite']
    else:
        N_ite = 10000
        simuParas['N_ite'] = N_ite       

    if 'max_re' in simuParas.keys():
        max_re = simuParas['max_re']
    else:
        max_re = 1e-9
        simuParas['max_re'] = max_re       
        
    if 'graphType' in simuParas.keys():
        graphType = simuParas['graphType']
    else:
        graphType = 'ER'
        simuParas['graphType'] = graphType
        
    if 'graphOptions' in simuParas.keys():
        graphOptions = simuParas['graphOptions']
    else:
        graphOptions = {}
        graphOptions['probIntra'] = 0.3 
        simuParas['graphOptions'] = graphOptions      
        
    if 'filterType' in simuParas.keys():
        filterType = simuParas['filterType']
    else:
        filterType = 'h'
        simuParas['filterType'] = filterType        

    if 'noiseType' in simuParas.keys():
        noiseType = simuParas['noiseType']
    else:
        noiseType = 'gaussion'
        simuParas['noiseType'] = noiseType  
        
    if 'noiseLevel' in simuParas.keys():
        noiseLevel = simuParas['noiseLevel']
    else:
        noiseLevel = 0
        simuParas['noiseLevel'] = noiseLevel          
        
    ## Results
    re_x = np.zeros(N_realiz)
    re_x_rc = np.zeros(N_realiz)    
    re_g = np.zeros(N_realiz)
    ite_list = np.zeros(N_realiz)
    max_re_matched_list = np.zeros(N_realiz)
    
    ## Experiment begins
    for n_realiz in range(N_realiz):
        print('n_realiz = ',n_realiz)
        G = SLOGtools.Graph(graphType, nNodes, graphOptions)
        G.computeGFT()
        d,An, eigenvalues, V   = SLOGtools.get_eig_normalized_adj(G.A)
        if 'filterType' == 'h':
            g0 = SLOGtools.h_generate_gso(nNodes,alpha, eigenvalues,L)
        else:
            g0 = SLOGtools.g_generate_gso(nNodes,alpha, eigenvalues,L)
        h0 = 1./g0
        X = SLOGtools.X_generate(nNodes,P,S)
        
        g0 = SLOGdata.to_numpy(g0)
        X = SLOGdata.to_numpy(X)
        V = SLOGdata.to_numpy(V)
    
        H = np.dot(V,np.dot(np.diag(h0),V.T))
        if noiseType == 'gaussion':
            noise = np.random.normal(0, 1, [nNodes,P])
            noise = noise/LA.norm(noise,'fro')*LA.norm(X,'fro')
        elif noiseType == 'uniform':
            noise = np.random.uniform(-1, 1, [nNodes,P])
            noise = noise/np.max(np.abs(noise))*np.max(np.abs(X))
        else:
            noise = np.zeros([nNodes,P])
            
        Y = np.dot(H,X) + noiseLevel*noise
        x = np.reshape(X.T,nNodes*P)
        x_hat,g_hat,n_ite,max_re_matched = SLOGtools.admm(Y,V,rho_0,eta_0,C,N_ite,max_re)
        x_hat = SLOGdata.to_numpy(x_hat)
        g_hat = SLOGdata.to_numpy(g_hat)
        
        # Recording results
        ite_list[n_realiz] = n_ite
        max_re_matched_list[n_realiz] = max_re_matched
        re_x[n_realiz] = LA.norm(x - x_hat)/LA.norm(x)
#         re_x_rc[n_realiz] = LA.norm(x - x_recv)/LA.norm(x)
        re_g[n_realiz] = LA.norm(g0 - g_hat)/LA.norm(g0)
    result = {}
    result['re_x'] = re_x
#     result['re_x_rc'] = re_x_rc    
    result['re_g'] = re_g 
    result['ite_list'] = ite_list
    result['max_re_matched_list'] = max_re_matched_list   
    result['simuParas'] = simuParas
    return result

def admmExperiment_single(nNodes, S, P, L,  **kwargs):
   
    ## Assertations
    assert nNodes >= S
    assert nNodes >= L
    
    ## Simulation parameters
    # simuParas should include:
    # alpha
    # rho_0
    # eta_0
    # C
    # N_ite
    # max_re    
    if 'simuParas' in kwargs.keys():
        simuParas = kwargs['simuParas']
    else:
        simuParas = {}
    
    if 'alpha' in simuParas.keys():
        alpha = simuParas['alpha']
    else:
        alpha = 1.0
        simuParas['alpha'] = alpha

    if 'rho_0' in simuParas.keys():
        rho_0 = simuParas['rho_0']
    else:
        rho_0 = 1.0
        simuParas['rho_0'] = rho_0

    if 'eta_0' in simuParas.keys():
        eta_0 = simuParas['eta_0']
    else:
        eta_0 = 1.0
        simuParas['eta_0'] = eta_0
        
    if 'C' in simuParas.keys():
        C = simuParas['C']
    else:
        C = nNodes
        simuParas['C'] = C
        
    if 'N_ite' in simuParas.keys():
        N_ite = simuParas['N_ite']
    else:
        N_ite = 10000
        simuParas['N_ite'] = N_ite       

    if 'max_re' in simuParas.keys():
        max_re = simuParas['max_re']
    else:
        max_re = 1e-9
        simuParas['max_re'] = max_re       
        
    if 'graphType' in simuParas.keys():
        graphType = simuParas['graphType']
    else:
        graphType = 'ER'
        simuParas['graphType'] = graphType
        
    if 'graphOptions' in simuParas.keys():
        graphOptions = simuParas['graphOptions']
    else:
        graphOptions = {}
        graphOptions['probIntra'] = 0.3 
        simuParas['graphOptions'] = graphOptions      
        
    if 'filterType' in simuParas.keys():
        filterType = simuParas['filterType']
    else:
        filterType = 'h'
        simuParas['filterType'] = filterType        

    if 'noiseType' in simuParas.keys():
        noiseType = simuParas['noiseType']
    else:
        noiseType = 'gaussion'
        simuParas['noiseType'] = noiseType  
        
    if 'noiseLevel' in simuParas.keys():
        noiseLevel = simuParas['noiseLevel']
    else:
        noiseLevel = 0
        simuParas['noiseLevel'] = noiseLevel          
        
    ## Results
    ## Experiment begins
    G = SLOGtools.Graph(graphType, nNodes, graphOptions)
    G.computeGFT()
    d,An, eigenvalues, V   = SLOGtools.get_eig_normalized_adj(G.A)
    if 'filterType' == 'h':
        g0 = SLOGtools.h_generate_gso(nNodes,alpha, eigenvalues,L)
    else:
        g0 = SLOGtools.g_generate_gso(nNodes,alpha, eigenvalues,L)
    h0 = 1./g0
    X = SLOGtools.X_generate(nNodes,P,S)
        
    g0 = SLOGdata.to_numpy(g0)
    X = SLOGdata.to_numpy(X)
    V = SLOGdata.to_numpy(V)
    
    H = np.dot(V,np.dot(np.diag(h0),V.T))
    if noiseType == 'gaussion':
        noise = np.random.normal(0, 1, [nNodes,P])
        noise = noise/LA.norm(noise,'fro')*LA.norm(X,'fro')
    elif noiseType == 'uniform':
        noise = np.random.uniform(-1, 1, [nNodes,P])
        noise = noise/np.max(np.abs(noise))*np.max(np.abs(X))
    else:
        noise = np.zeros([nNodes,P])  
    Y = np.dot(H,X) + noiseLevel*noise
    x = np.reshape(X.T,nNodes*P)
    x_hat,g_hat,n_ite,max_re_matched = SLOGtools.admm(Y,V,rho_0,eta_0,C,N_ite,max_re)
    x_hat = SLOGdata.to_numpy(x_hat)
    X_hat = x_hat.reshape([P,nNodes])
    X_hat = X_hat.T
    g_hat = SLOGdata.to_numpy(g_hat)
        # Recording results
    ite_list= n_ite
    max_re_matched_list= max_re_matched
    re_x = LA.norm(x - x_hat)/LA.norm(x)
    re_g = LA.norm(g0 - g_hat)/LA.norm(g0)
    result = {}
    result['re_x'] = re_x
    result['re_g'] = re_g 
    result['ite_list'] = ite_list
    result['X0'] = X
    result['Y'] = Y   
    result['g0'] = g0
    result['X_hat'] = X_hat
#     result['X_hat'] = X_hat   
    result['g_hat'] = g_hat 
    result['gso'] = G.A
    result['max_re_matched_list'] = max_re_matched_list   
    result['simuParas'] = simuParas
    return result

