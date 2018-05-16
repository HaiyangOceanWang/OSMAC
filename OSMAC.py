# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from scipy.optimize import minimize

def logistic_func(beta, x):
    return 1 / (1 + np.exp(-np.dot(x,beta)))

'''Functions for getting defferent Pi's'''
def pis_mVc(x,y,beta):
    '''
    rewrite mVc and mMSE,share the 'p' and 'dif'!!!!
    '''
    p=logistic_func(beta, x)
    dif=np.abs(y-p) 
    xnorm=np.linalg.norm(x,axis=1)
    pis=dif*xnorm
    pi=pis/sum(pis)
    return pi
def pis_mMSE1(x,y,beta):
    p=logistic_func(beta, x)
    dif=np.abs(y-p)
    w=p*(1-p)
    Mx=(x.T*w).dot(x)/y.shape[0]
    Mx_inv=np.linalg.solve(Mx)
    Mxx=x.dot(Mx_inv)
    Mxnorm=np.linalg.norm(Mxx,axis=1)
    pis=dif*Mxnorm
    pi=pis/sum(pis)
    return pi

def pis_mMSE(x,y,x_prop,pis_prop,beta):
    p=logistic_func(beta, x)
    
    p_prop=logistic_func(beta, x_prop)
    w_prop=p_prop*(1-p_prop)/pis_prop
    Mx=(x_prop.T*w_prop).dot(x_prop)
    
    dif=np.abs(y-p)
    Mx_inv=np.linalg.pinv(Mx)
    Mxx=x.dot(Mx_inv)
    Mxnorm=np.linalg.norm(Mxx,axis=1)
    pis=dif*Mxnorm
    pi=pis/sum(pis)
    return pi

def pis_LCC(x,y,beta):
    p=logistic_func(beta, x)
    dif=np.abs(y-p) 
    pi=dif/sum(dif)
    return pi
'''Subsample function'''
def subsample(x,y,r,pis):
    subsampledRows=np.random.choice(x.shape[0], r, p=pis)
    x_r=x[subsampledRows]
    y_r=y[subsampledRows]
    pis_r=pis[subsampledRows]
    return x_r,y_r,pis_r

'''Newton method for the normal mle'''
def mle(x,y):
    pis=np.ones(x.shape[0])
    return newton(x, y,pis)

'''Newton method for the weighted mle'''
def newton(x, y,pis, converge_change=.000001): 
    beta0 = np.repeat(0,x.shape[1])
    dist=1
    while(dist > converge_change):
        p = logistic_func(beta0,x)
        H=(x.T*(p*(1-p)/pis)).dot(x)
        J=np.sum(x.T*((p-y)/pis),axis=1)
        try:
            HJ=np.linalg.solve(H,J)
            beta1 = beta0 - HJ
        except:
            #H_inv=np.linalg.pinv(H);beta1 = beta0 - H_inv.dot(J)
            beta1 = beta0 - np.linalg.lstsq(H,J)[0]
            dist=np.linalg.norm(beta1-beta0)
            beta0=beta1
        else:
            dist=np.linalg.norm(beta1-beta0)
            beta0=beta1
    return beta1

def newton5(x, y,pis, converge_change=.000001): 
    beta0 = np.repeat(0,x.shape[1])
    dist=1
    while(dist > converge_change):
        p = logistic_func(beta0,x)
        H=(x.T*(p*(1-p)/pis)).dot(x)
        J=np.sum(x.T*((p-y)/pis),axis=1)
        #beta1 = beta0 - np.linalg.lstsq(H,J)[0]
        beta1 = beta0 - np.linalg.solve(H,J)
        #H_inv=np.linalg.pinv(H)
        #beta1 = beta0 - H_inv.dot(J)
        dist=np.linalg.norm(beta1-beta0)
        beta0=beta1
    return beta1

def newton1(x, y,pis, converge_change=.000001): 
    beta0 = np.repeat(0,x.shape[1])
    tau = 0.0
    dist=1
    while(dist > converge_change):
        p = logistic_func(beta0,x)
        H=(x.T*(p*(1-p)/pis)).dot(x)
        muk =np.power(np.linalg.norm(H),1+tau)
        Hr=H + muk*np.eye(x.shape[1])
        J=np.sum(x.T*((p-y)/pis),axis=1)
        #beta1 = beta0 - np.linalg.lstsq(H,J)[0]
        beta1 = beta0 - np.linalg.solve(Hr,J)
        #H_inv=np.linalg.pinv(H)
        #beta1 = beta0 - H_inv.dot(J)
        dist=np.linalg.norm(beta1-beta0)
        beta0=beta1
    return beta1

def newton3(x, y,pis):
    x0 = np.repeat(0,x.shape[1])
    maxk = 1e5
    n = np.shape(x0)[0]
    tau = 0.0
    k = 0
    epsilon = 1e-5

    while k < maxk:
        p = logistic_func(x0,x)
        gk = np.sum(x.T*((p-y)/pis),axis=1)     
        if  np.linalg.norm(gk) < epsilon:
            break
        muk = np.power(np.linalg.norm(gk),1+tau)
        Gk = (x.T*(p*(1-p)/pis)).dot(x)
        Ak = Gk + muk*np.eye(n)
        dk = -np.linalg.solve(Ak,gk)
        x0 += dk
        k += 1
    return x0

def newton4(x,y,pis):
    def log_likelihood(beta):
        score=np.dot(x,beta)
        l=np.sum((y*score - np.log(1 + np.exp(score)))/pis)
        
        return -l/x.shape[0]
    beta0=np.repeat(0,x.shape[1])
    o=minimize(log_likelihood,beta0,method='BFGS')
    return o.x

def two_steps(x,y,r0,r,method):
    n=y.shape[0]
    if method=='uni':
        pis_uni=np.repeat(1/n,n)
        x_uni,y_uni,pi_uni = subsample(x,y,r+r0,pis_uni)
        beta_uni=newton(x_uni,y_uni,pi_uni)
        result=beta_uni
    
    else:
        n1=np.count_nonzero(y)
        n0=n-n1
        pis_prop=y*(1/(2*n1))
        for count in range(y.shape[0]):
            if pis_prop[count]==0:
                pis_prop[count]=(1/(2*n0))
        x_prop,y_prop,pis_prop = subsample(x,y,r0,pis_prop)
        beta0 = newton(x_prop,y_prop,pis_prop)
    
        if method=='mvc':
            pi_mVc=pis_mVc(x,y,beta0)
            x_mVc,y_mVc,pismVc = subsample(x,y,r,pi_mVc)
            x1=np.append(x_prop,x_mVc,axis=0)
            y1=np.append(y_prop,y_mVc,axis=0)
            pis1=np.append(pis_prop,pismVc,axis=0)
            beta_mVc=newton(x1,y1,pis1)
            result=beta_mVc
        
        elif method=='mmse':
            pi_mMSE=pis_mMSE(x,y,x_prop,pis_prop,beta0)
            #pi_mMSE=pis_mMSE(x,y,beta0)
            x_mMSE,y_mMSE,pismMSE = subsample(x,y,r,pi_mMSE)
            x1=np.append(x_prop,x_mMSE,axis=0)
            y1=np.append(y_prop,y_mMSE,axis=0)
            pis1=np.append(pis_prop,pismMSE,axis=0)
            beta_mMSE=newton(x1,y1,pis1)
            result=beta_mMSE
        
        elif method=='lcc':
            pi_LCC=pis_LCC(x,y,beta0)
            x_LCC,y_LCC,pisLCC = subsample(x,y,r,pi_LCC)
            beta_LCC=mle(x_LCC,y_LCC)+beta0
            result=beta_LCC
    
    return result
