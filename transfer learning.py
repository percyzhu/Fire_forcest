# Setup GPU for training (use tensorflow v2.7)
import os
import tensorflow as tf
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use '-1' for CPU only, use '0' if there is only 1 GPU
                                          # use GPU number for multiple GPUs
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.75   ###调整
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session
import numpy as np
import gc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.io
from tensorflow import keras, transpose
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Input, Concatenate, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import  Activation, CuDNNLSTM,  RepeatVector
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import time
import joblib
from keras.models import load_model

# fonts
font = {'family':'Times New Roman'}
matplotlib.rc('font',**font)
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['Times New Roman']  

# Load data
dataDir = 'D:/temperature prediction/'  # Replace the directory
mat = scipy.io.loadmat(dataDir+'FDS_dataset.mat')
NT=30
X_data = mat['input_tf'][:,:,0:NT]
index_data=mat['input_tf'][:,:,NT]
y_data = mat['target_tf']
geo_data = mat['input_jhe']
acute_data = tf.zeros([160, 4])
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

acute_train = acute_data[0:len(train_indices[0])]
acute_test = acute_data[len(train_indices[0]):]
# Unknown data
X_pred = mat['input_pred_tf'][:,:,0:NT]
index_pred=mat['input_pred_tf'][:,:,NT]
y_pred_ref = mat['target_pred_tf']
geo_pred = mat['input_pred_jhe']
acute_pred = tf.zeros([40, 4])

# Scale data
X_pred_flatten = np.reshape(X_pred, [X_pred.shape[0]*X_pred.shape[1], X_pred.shape[2]])
X_pred_flatten_map = scaler_X.transform(X_pred_flatten)
X_pred_map = np.reshape(X_pred_flatten_map, [X_pred.shape[0], X_pred.shape[1], X_pred.shape[2]])

y_pred_ref_flatten = np.reshape(y_pred_ref, [y_pred_ref.shape[0]*y_pred_ref.shape[1], y_pred_ref.shape[2]])
y_pred_ref_flatten_map = scaler_y.transform(y_pred_ref_flatten)
y_pred_ref_map = np.reshape(y_pred_ref_flatten_map, [y_pred_ref.shape[0], y_pred_ref.shape[1], y_pred_ref.shape[2]])

geo_pred_flatten = np.reshape(geo_pred, [geo_pred.shape[0], geo_pred.shape[1]])
geo_pred_flatten_map = scaler_geo.transform(geo_pred_flatten)
geo_pred_map = np.reshape(geo_pred_flatten_map, [geo_pred.shape[0], geo_pred.shape[1]])

X_pred = X_pred_map
y_pred_ref = y_pred_ref_map
y_pred = y_pred_ref
geo_pred = geo_pred_map 
#y_pred = np.reshape(y_pred_ref, [y_pred_ref.shape[0],  y_pred_ref.shape[2]])

data_dim = X_train.shape[2]     # number of input gas temperatures
jhe_dim = geo_train.shape[1]    # number of geometry parameters
timesteps = X_train.shape[1]    # fire exposure time
num_classes = y_train.shape[2]  # number of output fire states
batch_size = 64               

# Delete variables not in use
del X_data_flatten,X_data_flatten_map,y_data_flatten,y_data_flatten_map,X_data_map
del y_data_map,X_pred_flatten,X_pred_flatten_map,X_pred_map,y_pred_ref_flatten,y_pred_ref_flatten_map
del y_pred_ref_map

def create_data_seq(seq1,seq2,seq3,seq4,seq5,windowsize1,windowsize2):
    x=[]
    y=[]
    z=[]
    geo=[]
    index=[]
    L=seq1.shape[1]
    F=seq1.shape[0]
    for j in range(F):
        for i in range(L-windowsize1-windowsize2):
            if (((i+windowsize1-1)>seq4[j,0]) | ((i+windowsize1+windowsize2-2)<seq4[j,0])) & \
                ((i+windowsize1-1>seq4[j,1]) | (i+windowsize1+windowsize2-2<seq4[j,1])) & \
                ((i+windowsize1-1>seq4[j,2]) | (i+windowsize1+windowsize2-2<seq4[j,2])) & \
                ((i+windowsize1-1>seq4[j,3]) | (i+windowsize1+windowsize2-2<seq4[j,3])):
                x_w=np.expand_dims(seq1[j,i:i+windowsize1,:],axis=0)
                y_w=np.expand_dims(seq2[j,i+windowsize1-1,:],axis=0)
                z_w=np.expand_dims(seq1[j,i+windowsize1-1:i+windowsize1+windowsize2,0:30],axis=0)
                geo_w=np.expand_dims(seq3[j],axis=0)
                index_w=np.expand_dims(seq5[j,i:i+windowsize1],axis=0)
                x.extend(x_w)
                y.extend(y_w)
                z.extend(z_w)
                geo.extend(geo_w)
                index.extend(index_w)
    return np.array(x), np.array(y), np.array(z), np.array(geo), np.array(index)

windowsize1=5
windowsize2=5
trainX,trainY,trainZ,trainJ,trainindex=create_data_seq(X_train,y_train,geo_train,acute_train,index_train,windowsize1,windowsize2)
testX, testY, testZ, testJ, testindex =create_data_seq(X_test, y_test, geo_test, acute_test, index_test, windowsize1,windowsize2)
predX, predY, predZ, predJ, predindex =create_data_seq(X_pred, y_pred, geo_pred, acute_pred, index_pred, windowsize1,windowsize2)

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
# Neural Network model

quantiles=[0.05,0.5,0.8,0.95]
adam = Adam(learning_rate=0.0001, decay=0.001)
# Trainfer learning
# Load the pre-trained model
model = load_model(dataDir + 'results/best_model_pretrain.h5', custom_objects={'self_loss1': self_loss1})
for layer in model.layers:
    layer.trainable = False
model.get_layer('Dense2').trainable = True
model.get_layer('Dense3').trainable = True
model.summary()                                          

best_loss = 10
train_loss = []
test_loss = []
history = []
Reg_loss = []
Fen_loss = []
Cla_loss = []
pred_loss = []


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    figsize = 9,6
    figure, ax = plt.subplots(figsize=figsize)
    plt.figure()
    plt.yscale("log")
    plt.xlabel('episode number')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], 'k-',label='Train loss',alpha=0.5)
    plt.plot(hist['epoch'], hist['val_loss'],'r-',label='Test loss',alpha=0.5)
    plt.legend(frameon=False,fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.tick_params(axis='both', which='major', tickdir='in', length=5)
    ax.tick_params(axis='both', which='minor', tickdir='in', length=3)
    plt.margins(0.05)
    plt.subplots_adjust(top=0.95,bottom=0.15,left=0.12,right=0.95,hspace=0,wspace=0)
    plt.savefig(r'results\transfer learning.svg')
    plt.show()

# Training
with tf.device('/device:GPU:0'):
    start = time.time()
    call1=tf.keras.callbacks.ModelCheckpoint(filepath=dataDir + 'results/best_model_trainfer.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1, save_freq="epoch"),    
    trainY_Reg = trainY[:,0:2]
    trainY_Fen = trainY[:,2:3]
    testY_Reg = testY[:,0:2]
    testY_Fen = testY[:,2:3]

    history=model.fit([trainX, trainindex, trainJ], [trainY_Reg, trainY_Fen, trainZ],
              batch_size=batch_size,
              validation_data=([testX, testindex, testJ], [testY_Reg, testY_Fen, testZ]),
              shuffle=True,
              use_multiprocessing=True,
              callbacks=[call1],
              workers=12,     
              epochs=500) 
                                  
    end = time.time()
    running_time = (end - start)/3600
    print('Running Time: ', running_time, ' hour')
plot_history(history)
plt.close()