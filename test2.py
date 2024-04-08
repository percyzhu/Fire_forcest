# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:11:05 2023

@author: admin
"""
# Setup GPU for training (use tensorflow v2.7)
import os
import tensorflow as tf
import pandas as pd
from keras.models import load_model
import scipy.io
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
tf.compat.v1.keras.backend.set_session
import numpy as np
import gc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm
import scipy.io
from tensorflow import keras, sign,transpose
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Input, Concatenate, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.compat.v1.keras.layers import LSTM, Activation, CuDNNLSTM, Conv1D, TimeDistributed, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import random, time
dataDir = 'D:/temperature prediction/'  # Replace the directory
mat = scipy.io.loadmat(dataDir+'test2.mat')
# Unknown data
X_pred = mat['input_pred_tf'][:,:,0:30]
index_pred=mat['input_pred_tf'][:,:,30]
geo_pred = np.expand_dims(np.array([12,2,6,7]),axis=0)

font1={'family':'Times New Roman','weight':'normal','size':13}

def self_loss1(y_true, y_pred):
    loss=[]
    quantiles=[0.05,0.5,0.8,0.95]
    for i,q in enumerate(quantiles):
        error=tf.subtract(y_true,y_pred[:,:,:,i])
        loss_q=tf.reduce_mean(tf.maximum(q*error,(q-1)*error))
        loss.append(loss_q)
    L=tf.convert_to_tensor(loss)
    totalloss=tf.reduce_mean(L)
    return totalloss
model_best = load_model(dataDir + 'best_model_trainfer.h5', custom_objects={'self_loss1': self_loss1} )
windowsize1=5
windowsize2=5

middleX=X_pred.copy()
for num in range(1):       
    for start0 in range(11):
        start=(start0+1)*5+1
        pred_Tall=[]
        ###identify damaged fire thermocouples
        #Spearman correlation
        [relation,zero]=scipy.stats.spearmanr(X_pred[num,0:start+windowsize1,:])
        #relation=np.corrcoef(transpose(X_pred[num,start-windowsize1:start+windowsize1,:]))
        relation_S=np.sum(relation,axis=1)/len(relation)
        Pfailindex1=np.squeeze(np.where(relation_S<0.6))
        #standard deviation
        bzcha=np.std(X_pred[num,start-windowsize1:start+windowsize1,:],axis=0)#start-windowsize1
        meanbzc=np.mean(bzcha)
        bzcbzcha=np.std(bzcha)
        Pfailindex2=np.squeeze(np.where((bzcha<meanbzc-3*bzcbzcha)|(bzcha>meanbzc+3*bzcbzcha)))
        ###
        Pfailindex=np.union1d(Pfailindex1,Pfailindex2)   # damaged thermocouple
        Pworkindex=np.setdiff1d(range(0, 30),Pfailindex) # normal thermocouple
        # fix the damaged thermocouple
        geo=geo_pred[0]
        x0=[-geo[0]*0.5,-geo[0]*0.375,-geo[0]*0.125,geo[0]*0.125,geo[0]*0.375,geo[0]*0.5]
        y0=[0,geo[1],-geo[1],geo[1]*2,-geo[1]*2]
        X,Y=np.meshgrid(x0,y0)
        X=np.squeeze(np.reshape(X,[1,30]))
        Y=np.squeeze(np.reshape(Y,[1,30]))
        if len(Pworkindex)>0:
            Xwork=X[Pworkindex]
            Ywork=Y[Pworkindex]
            Xfail=X[Pfailindex]
            Yfail=Y[Pfailindex]
            if start0>0:               
                for j in range(windowsize1):
                    Z=np.squeeze(np.reshape(X_pred[num,start+j,:],[1,30]))
                    Zwork= Z[Pworkindex]
                    Zfail= Z[Pfailindex]   
                    f = interpolate.interp1d(Pworkindex, Zwork, kind='linear',fill_value='extrapolate')
                    Znew = f(Pfailindex) 
                    for failnum in range(len(Pfailindex)):   
                        middleX[num,start+j,Pfailindex[failnum]]=Znew[failnum]
            else:
                for j in range(2*windowsize1):
                    Z=np.squeeze(np.reshape(X_pred[num,start-windowsize1+j,:],[1,30]))
                    Zwork= Z[Pworkindex]
                    Zfail= Z[Pfailindex]   
                    f = interpolate.interp1d(Pworkindex, Zwork, kind='linear',fill_value='extrapolate')
                    Znew = f(Pfailindex) 
                    for failnum in range(len(Pfailindex)):   
                        middleX[num,start-windowsize1+j,Pfailindex[failnum]]=Znew[failnum]#,failnum
X_pred = middleX.copy()

# Scale data
scaler_X = joblib.load(dataDir+'scaler_Xscheme.save')
scaler_y = joblib.load(dataDir+'scaler_yscheme.save')
scaler_geo = joblib.load(dataDir+'scaler_geoscheme.save')

X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0]*X_pred.shape[1], X_pred.shape[2]])
X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]])
X_pred = X_pred_map

geo_pred_flatten = np.reshape(geo_pred, [geo_pred.shape[0], geo_pred.shape[1]])
geo_pred_flatten_map = scaler_geo.transform(geo_pred_flatten)
geo_pred_map = np.reshape(geo_pred_flatten_map, [geo_pred.shape[0], geo_pred.shape[1]])
geo_pred = geo_pred_map 

def create_data_seq(seq1,seq3,seq5,windowsize1,windowsize2):
    x=[]
    y=[]
    z=[]
    geo=[]
    index=[]
    L=seq1.shape[1]
    F=seq1.shape[0]
    for j in range(F):
        for i in range(L-windowsize1-windowsize2):
            x_w=np.expand_dims(seq1[j,i:i+windowsize1,:],axis=0)
            z_w=np.expand_dims(seq1[j,i+windowsize1-1:i+windowsize1+windowsize2,0:30],axis=0)
            geo_w=np.expand_dims(seq3[j],axis=0)
            index_v=np.expand_dims(seq5[j,i:i+windowsize1],axis=0)
            index_w=np.expand_dims(index_v,axis=2)
            x.extend(x_w)
            z.extend(z_w)
            geo.extend(geo_w)
            index.extend(index_w)
    return np.array(x), np.array(z), np.array(geo), np.array(index)
predX_orig,  predZ_orig, predJ_orig, predindex =create_data_seq(X_pred, geo_pred, index_pred, windowsize1,windowsize2)


data_dim=30
quantiles=[0.05,0.5,0.8,0.95]
number=X_pred.shape[0]
length=X_pred.shape[1]-windowsize1-windowsize2
number=1
quantilepre_all=[]
PINA_all=[]
PINR_all=[]
numPIPC=[]
fig,ax=plt.subplots()
failratio=0.1
X_pred = middleX.copy()
time_start=time.time()
for xuhao in range(1): 
    num=0
    quantilepre=[]
    PINA=[]    
    PINR=[] 
    singlePIPC=[]
    predstate=[]
    realstate=[]
    for start0 in range(1):#9
        start=35#(start0+1)*5
        pred_Tall=[]
        for f in range(len(quantiles)):
            pred_T0=[]
            x   =np.expand_dims(predX_orig[num*length + start,:,:],axis=0)
            index =np.expand_dims(predindex[num*length + start,:],axis=0)
            geo=np.expand_dims(predJ_orig[num*length + start,:],axis=0)
            curve=X_pred[num,0:start+windowsize1,:]
            k=int(np.floor((length-start)/windowsize2)) 
            for ts in range(k+1):#
                history=scaler_X.inverse_transform(np.reshape(x,[-1,data_dim]))
                y_pure_preds = model_best.predict([x, index, geo])  
                x=y_pure_preds[2]
                FFx=np.reshape(x,[windowsize2+1,data_dim,len(quantiles)])
                Fx0=FFx[:,:,f]
                T0=scaler_X.inverse_transform(Fx0)
                deltaT=T0[0,0:data_dim]-history[windowsize2-1,0:data_dim]                
                T0=T0-deltaT
                nextT=T0[1:windowsize2+1,:]
                pred_T0.extend(nextT)
                scalex=scaler_X.transform(nextT)
                x=np.reshape(scalex,[1,windowsize2,data_dim])         
                if ts ==0:
                    rp=y_pure_preds[0]
                    rq=y_pure_preds[1]
                    ry=scaler_y.inverse_transform(np.hstack([rp,rq]))
                # curve=np.concatenate((curve,T0), axis=0)
                # curve_smooth = gaussian_filter(curve,sigma=0.2)
                # T0=curve_smooth[(start+windowsize1+ts*windowsize2):(start+windowsize1+(ts+1)*windowsize2),:]
                # pred_T0.extend(T0)
                # x=scaler_X.transform(T0)
                # x=np.reshape(x,[1,windowsize2,data_dim])                   
            pred_T=np.array(pred_T0)
            pred_Tall.append(pred_T)
    #     deltatime=20        
    #     time=np.arange(start+windowsize1,start+windowsize1+20,1)
    #     for f in range(len(quantiles)):
    #         a=pred_Tall[f]
    #         pred_Tall[f]=a[0:deltatime,:]#60-start-windowsize1
    #     #EVALUATION INDEX   
    #     time=np.arange(start+windowsize1,start+windowsize1+deltatime,1)
    #     for f in range(len(quantiles)):
    #         a=pred_Tall[f]
    #         pred_Tall[f]=a[0:deltatime,:]#60-start-windowsize1
    #     #EVALUATION INDEX
    #     real_Tall=X_pred[num,start+windowsize1:start+windowsize1+deltatime,:]    
    #     singlepre=np.array(sign(sign(pred_Tall-real_Tall+10)+1))
    #     mid=np.expand_dims(np.sum(np.sum(singlepre,axis=1),axis=1)/(singlepre.shape[1]*singlepre.shape[2]),axis=0)
    #     singlePIPC.extend(mid)
    #     quantilepre.append(singlepre)
    #     IW=pred_Tall-real_Tall
    #     RS=IW/real_Tall
    #     PINA.append(IW)
    #     PINR.append(RS)
    #     # time=np.arange(start+windowsize1,60,1)
    #     # plt.figure(num*4+start+1)
    #     # plt.plot(time,pred_Tall[0][:,xuhao],color='green',linestyle='dashed',label=str(quantiles[0]*100)+'%')
    #     # plt.plot(time,pred_Tall[1][:,xuhao],color='purple',linestyle='dashed',label=str(quantiles[1]*100)+'%')
    #     # plt.plot(time,pred_Tall[2][:,xuhao],color='orange',linestyle='dashed',label=str(quantiles[2]*100)+'%')
    #     # plt.plot(time,pred_Tall[3][:,xuhao],color='red',linestyle='dashed',label=str(quantiles[3]*100)+'%')       
    #     # plt.fill_between(time,pred_Tall[0][:,xuhao],pred_Tall[3][:,xuhao],facecolor='#B0C4DE',alpha=0.6)
    #     # time0=np.arange(0,60,1)
    #     # real_Tall=X_pred[num,0:60,:]    
    #     # plt.plot(time0,real_Tall[:,xuhao],color='black',linestyle='solid',label='real T')
    #     # plt.legend(loc='upper left', prop=font1)
    #     # plt.axis([0, 60, 0, 1200])
    #     # plt.xlabel('t / min', fontsize=13)
    #     # plt.ylabel('T / ℃', fontsize=13)#font={'family':'Times New Roman', 'size':16}
    #     # labels=ax.get_xticklabels()+ax.get_yticklabels()
    #     # [label.set_fontname('Times New Roman') for label in labels]
    #     # plt.tick_params(labelsize=13)
    #     # plt.savefig(str(xuhao)+'-'+str(start)+'.jpg',dpi=150)
    #     # plt.clf()
    #     # plt.close()
    # numPIPC.append(np.array(singlePIPC))    
    # quantilepre_all.append(quantilepre)
    # PINA_all.append(PINA)  
    # PINR_all.append(PINR)
# a=np.array(numPIPC)
# PIPC=np.sum(a,axis=0)/(a.shape[0])
time_end=time.time()
deltatime=time_end-time_start
# for f in range(len(quantiles)): 
#     data1=[]
#     data2=[]
#     for start0 in range(1):  
#         start=15#start0*5
#         data3=[]
#         data4=[]
#         for num in range(number):   
#             data3.append(PINA_all[num][start0][f,0:deltatime,:])  
#             data4.append(PINR_all[num][start0][f,0:deltatime,:])  
#         index_reliable=np.squeeze(np.array(data3).reshape(-1,1))
#         data1.append(index_reliable)
#         index_accuracy=np.squeeze(np.array(data4).reshape(-1,1))
#         data2.append(index_accuracy)        
#     plt.boxplot(data1,whis=(5,95),medianprops={'color':'red','linewidth':'1.5'},meanline=True,showmeans=True,showfliers=False)
#     plt.yticks(np.arange(-200,200,50))
#     plt.savefig(str(f)+'-fix.jpg',dpi=150)
#     plt.show()