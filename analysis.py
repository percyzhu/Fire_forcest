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
from tensorflow import keras, transpose, sign
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Input, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import Activation, CuDNNLSTM, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
dataDir = 'D:/temperature prediction/'  # Replace the directory
mat = scipy.io.loadmat(dataDir+'classicalfire_dataset.mat')

# Unknown data
X_pred = mat['input_pred_tf'][:,:,0:30]
index_pred=mat['input_pred_tf'][:,:,30]
y_pred_ref = mat['target_pred_tf']
geo_pred = mat['input_pred_jhe']

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

X_data = mat['input_tf'][:,:,0:30]
index_data=mat['input_tf'][:,:,30]
y_data = mat['target_tf']
geo_data = mat['input_jhe']

train_indices = mat['trainInd'] - 1
test_indices = mat['valInd'] - 1

# Scale data
scaler_X = joblib.load(dataDir+'scaler_Xscheme.save')
scaler_y = joblib.load(dataDir+'scaler_yscheme.save')
scaler_geo = joblib.load(dataDir+'scaler_geoscheme.save')
X_data_flatten = np.reshape(X_data, [X_data.shape[0]*X_data.shape[1], X_data.shape[2]])

X_data_flatten_map = scaler_X.transform(X_data_flatten)
X_data_map = np.reshape(X_data_flatten_map, [X_data.shape[0], X_data.shape[1], X_data.shape[2]])

y_data_flatten = np.reshape(y_data, [y_data.shape[0]*y_data.shape[1], y_data.shape[2]])

y_data_flatten_map = scaler_y.transform(y_data_flatten)
y_data_map = np.reshape(y_data_flatten_map, [y_data.shape[0], y_data.shape[1], y_data.shape[2]])

geo_data_flatten = np.reshape(geo_data, [geo_data.shape[0], geo_data.shape[1]])

geo_data_flatten_map = scaler_geo.transform(geo_data_flatten)
geo_data_map = np.reshape(geo_data_flatten_map, [geo_data.shape[0], geo_data.shape[1]])

X_data_new = X_data_map
y_data_new = np.squeeze(y_data_map)
geo_new = geo_data_map


X_train = X_data_new[0:len(train_indices[0])]
y_train = y_data_new[0:len(train_indices[0])]
index_train=index_data[0:len(train_indices[0])]
X_test = X_data_new[len(train_indices[0]):]
y_test = y_data_new[len(train_indices[0]):]
index_test=index_data[len(train_indices[0]):]
geo_train = geo_new[0:len(train_indices[0])]
geo_test = geo_new[len(train_indices[0]):]

X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0]*X_pred.shape[1], X_pred.shape[2]])
X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]])
y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0]*y_pred_ref.shape[1], y_pred_ref.shape[2]])
y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])
X_pred = X_pred_map
y_pred = y_pred_ref_map  


geo_pred_flatten = np.reshape(geo_pred, [geo_pred.shape[0], geo_pred.shape[1]])
geo_pred_flatten_map = scaler_geo.transform(geo_pred_flatten)
geo_pred_map = np.reshape(geo_pred_flatten_map, [geo_pred.shape[0], geo_pred.shape[1]])
geo_pred = geo_pred_map 

def create_data_seq(seq1,seq2,seq3,seq5,windowsize1,windowsize2):
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
            y_w=np.expand_dims(seq2[j,i+windowsize1-1,:],axis=0)
            z_w=np.expand_dims(seq1[j,i+windowsize1-1:i+windowsize1+windowsize2,0:30],axis=0)
            geo_w=np.expand_dims(seq3[j],axis=0)
            index_v=np.expand_dims(seq5[j,i:i+windowsize1],axis=0)
            index_w=np.expand_dims(index_v,axis=2)
            x.extend(x_w)
            y.extend(y_w)
            z.extend(z_w)
            geo.extend(geo_w)
            index.extend(index_w)
    return np.array(x), np.array(y), np.array(z), np.array(geo), np.array(index)
predX_orig, predY_orig, predZ_orig, predJ_orig, predindex =create_data_seq(X_pred, y_pred, geo_pred, index_pred, windowsize1,windowsize2)
trainX_orig, trainY_orig, trainZ_orig, trainJ_orig, trainindex =create_data_seq(X_train, y_train, geo_train, index_train, windowsize1,windowsize2)


data_dim=30
quantiles=[0.05,0.5,0.8,0.95]
number=X_pred.shape[0]
length=X_pred.shape[1]-windowsize1-windowsize2
number=30
quantilepre_all=[]
PINA_all=[]
numPIPC=[]
fig,ax=plt.subplots()
failratio=0.1
X_pred = mat['input_pred_tf'][:,:,0:30]
for num in range(number):  
    quantilepre=[]
    PINA=[]    
    singlePIPC=[]
    for start0 in range(3):
        start=(start0+1)*15-5 # fire exposure time
        pred_Tall=[]               
        for f in range(len(quantiles)):
            pred_T0=[]
            x   =np.expand_dims(predX_orig[num*length + start,:,:],axis=0)
            index =np.expand_dims(predindex[num*length + start,:],axis=0)
            geo=np.expand_dims(predJ_orig[num*length + start,:],axis=0)
            curve=X_pred[num,0:start+windowsize1,:]
            k=int(np.floor((length-start)/windowsize2)) 
            for ts in range(k+1):
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
            pred_T=np.array(pred_T0)
            pred_Tall.append(pred_T)                
        time=np.arange(start+windowsize1,start+windowsize1+20,1)
        for f in range(len(quantiles)):
            a=pred_Tall[f]
            pred_Tall[f]=a[0:20,:] # when time prediction interval = 20
        #EVALUATION INDEX
        real_Tall=X_pred[num,start+windowsize1:start+windowsize1+20,:]    
        singlepre=np.array(sign(sign(pred_Tall-real_Tall)+1))
        mid=np.expand_dims(np.sum(np.sum(singlepre,axis=1),axis=1)/(singlepre.shape[1]*singlepre.shape[2]),axis=0)
        singlePIPC.extend(mid)
        quantilepre.append(singlepre)
        WS=pred_Tall-real_Tall
        PINA.append(WS)
        ## plot curve
        # xuhao=2
        # time=np.arange(start+windowsize1,60,1)
        # plt.figure(num*4+start+1)
        # plt.plot(time,pred_Tall[0][:,xuhao],color='green',linestyle='dashed',label=str(quantiles[0]*100)+'%')
        # plt.plot(time,pred_Tall[1][:,xuhao],color='purple',linestyle='dashed',label=str(quantiles[1]*100)+'%')
        # plt.plot(time,pred_Tall[2][:,xuhao],color='orange',linestyle='dashed',label=str(quantiles[2]*100)+'%')
        # plt.plot(time,pred_Tall[3][:,xuhao],color='red',linestyle='dashed',label=str(quantiles[3]*100)+'%')       
        # plt.fill_between(time,pred_Tall[0][:,xuhao],pred_Tall[3][:,xuhao],facecolor='#B0C4DE',alpha=0.6)
        # time0=np.arange(0,61,1)
        # real_Tall=X_pred[num,0:61,:]    
        # plt.plot(time0,real_Tall[:,xuhao],color='black',linestyle='solid',label='real T')
        # plt.legend(loc='upper left', prop=font1)
        # plt.axis([0, 60, 0, 1000])
        # plt.xlabel('t / min', fontsize=13)
        # plt.ylabel('T / ℃', fontsize=13)
        # labels=ax.get_xticklabels()+ax.get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]
        # plt.tick_params(labelsize=13)
        # plt.savefig(str(num)+'-'+str(start)+'.jpg',dpi=150)
        # plt.clf()
        # plt.close()
    numPIPC.append(np.array(singlePIPC))    
    quantilepre_all.append(quantilepre)
    PINA_all.append(PINA) 
a=np.array(numPIPC)
PIPC=np.sum(a,axis=0)/(a.shape[0])
## evaluation index
for f in range(len(quantiles)):
    data1=[]
    for start0 in range(8):  
        start=start0*5
        data=[]
        for num in range(number):   
            data.append(PINA_all[num][start0][f,0:20,:])  # when fire prediction interval=20
        data0=np.squeeze(np.array(data).reshape(-1,1))
        data1.append(data0)
    plt.boxplot(data1,whis=(25,75),medianprops={'color':'red','linewidth':'1.5'},meanline=True,showmeans=True,showfliers=False)
    plt.savefig(str(f)+'-fix.jpg',dpi=150)
    plt.show()    