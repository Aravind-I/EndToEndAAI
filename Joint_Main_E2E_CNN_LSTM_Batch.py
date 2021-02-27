
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io
import os
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint ,EarlyStopping
from keras.layers.wrappers import Bidirectional
from keras.utils.generic_utils import Progbar
from keras.layers.normalization import BatchNormalization
#from keras.utils.visualize_util import plot
from keras.layers import LSTM, Dropout, GRU, Convolution1D,  MaxPooling1D, Flatten,Reshape
from keras.layers import Input, Dense, Dropout, TimeDistributed, GlobalAveragePooling1D
import sys
import numpy
import librosa
import tensorflow as tf
import HTK
from keras.preprocessing.sequence import pad_sequences
#http://philipperemy.github.io/keras-stateful-lstm/


# In[2]:


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def Relu_log(In):
    Out_relu=tf.nn.relu(np.abs(In))
    Out_log=tf.log(Out_relu+0.01)
    return Out_log

Wav_Fs=8000
## Raw waveform
window_size=int(25*0.001*Wav_Fs) #500 window len
Hop_len=int(10*0.001*Wav_Fs) #160 wind shift
std_frac=0.25
n_mfcc=13

nb_outputs=12 #pred EMA dim

### CNN Spec
#CNN1_filters=100
CNN2_filters=40
CNN1_flength=int(20*0.001*Wav_Fs)
CNN2_flength=8
inputDim=1

NoUnits=150 #LSTM units
BatchSize=5
NoEpoch=50
htkfile = HTK.HTKFile()
H_window = scipy.signal.hann(window_size, sym=False)


# In[3]:



def Get_Wav_EMA_PerFile(EMA_file,Wav_file,F):
    EmaMat=scipy.io.loadmat(EmaDir+EMA_file);
    EMA_temp=EmaMat['EmaData'];
    EMA_temp=np.transpose(EMA_temp)# time X 18
    Ema_temp2=np.delete(EMA_temp, [4,5,6,7,10,11],1) # time X 12
    MeanOfData=np.mean(Ema_temp2,axis=0) 
    Ema_temp2-=MeanOfData
    C=np.sqrt(np.mean(np.square(Ema_temp2),axis=0))
    Ema=Ema_temp2#np.divide(Ema_temp2,C) # Mean remov & var normailized
    [aE,bE]=Ema.shape
    
    sig, sr = librosa.load(WavDir+Wav_file, sr=Wav_Fs);
    sig = sig/max(abs(sig))
    #dither = 1e-6*np.random.rand(sig.shape[0])
    y = preemphasis(sig)# + dither
    y = (y-np.mean(y))/np.std(y) 
    y_framed = librosa.util.frame(y,window_size, Hop_len).astype(np.float64).T
    y_framed*=H_window#y_framed=np.transpose(y_framed) # T X 320
    
    #y_framed=librosa.util.normalize(y_framed,norm=2,axis=1)
#     MeanOfData_Wav=np.mean(y_framed,axis=0) 
#     y_framed-=MeanOfData_Wav
#     C_W=np.sqrt(np.mean(np.square(y_framed),axis=0))
#     y_framed_N=np.divide(y_framed,C_W)
    
    [aW,bW]=y_framed.shape
    tEnd=np.min([aW,aE])

    #print F.type
    EBegin=np.int(BeginEnd[0,F]*100)
    EEnd=np.int(BeginEnd[1,F]*100)
    
    ### MFCC ###
    htkfile.load(MFCCpath+Wav_file[:-4]+'.mfc')
    feats = np.asarray(htkfile.data)
    mean_G = np.mean(feats, axis=0)
    std_G = np.std(feats, axis=0)
    feats = std_frac*(feats-mean_G)/std_G
    MFCC_G = feats
    TimeStepsTrack=EEnd-EBegin
    return Ema[EBegin:EEnd,:], y_framed[EBegin:EEnd,:], MFCC_G[EBegin:EEnd,:n_mfcc],TimeStepsTrack # with out silence
    #return Ema[0:tEnd,:], np.atleast_3d(y_framed[0:tEnd,:]) # with silence


# In[4]:
X_valseq=[];Youtval=[];
X_trainseq=[];Youttrain=[];
X_testseq=[];Youttest=[];
TT_Test=[];TT_Train=[];TT_Valid=[]

for CNN1_filters in [40, 100, 256]
    Set1=['M1', 'M2' , 'M3','M4','F1', 'F2','F3','F5']
    OutDir='E2E_Results_Batch/'
    RootDir='../../../SPIRE_EMA/'
    for ss in np.arange(0,len(Set1)):
        Sub=Set1[ss]#'Anand_S'
        print(Sub)
        WavDir=RootDir+'DataBase/'+Sub+'/Neutral/WavClean/';
        EmaDir=RootDir+'DataBase/'+Sub+'/Neutral/EmaClean/';
        BeginEndDir=RootDir+'/StartStopMat/'+Sub+'/';
        MFCCpath=RootDir+'/DataBase/'+Sub+'/Neutral/MfccHTK/'

        EMAfiles=sorted(os.listdir(EmaDir))
        Wavfiles=sorted(os.listdir(WavDir))
        StartStopFile=os.listdir(BeginEndDir)
        StartStopMAt=scipy.io.loadmat(BeginEndDir+StartStopFile[0])
        BeginEnd=StartStopMAt['BGEN']
        #window_size=500

        F=5 # Fold No
        for i in np.arange(0,460):
            if  (((i+F)%10)==0):# Test
                E_t,W_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
                W_t=W_t[np.newaxis,:,:,np.newaxis]
                E_t=E_t[np.newaxis,:,:]
                M_t=M_t[np.newaxis,:,:]
                Youttest.append(E_t)
                X_testseq.append(W_t)
                TT_Test.append(TT)
                #print('Test '+str(i))
            elif (((i+F+1)%10)==0):# Validation
                E_t,W_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
                W_t=W_t#[np.newaxis,:,:,:]
                E_t=E_t#[np.newaxis,:,:]
                M_t=M_t#[np.newaxis,:,:]
                Youtval.append(E_t)
                X_valseq.append(W_t)
                TT_Valid.append(TT)
            else: # Train 
                E_t,W_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
                W_t=W_t#[np.newaxis,:,:,:]
                E_t=E_t#[np.newaxis,:,:]
                M_t=M_t#[np.newaxis,:,:]
                Youttrain.append(E_t)
                X_trainseq.append(W_t)
                TT_Train.append(TT)

        #https://github.com/keras-team/keras/issues/9788
    TT_Total=(np.concatenate([np.array(TT_Test),np.array(TT_Valid),np.array(TT_Train)]))
    TT_max=400#np.max(TT_Total)

    #padded = pad_sequences(sequences, padding='post',maxlen=TT_max)
    X_valseq=pad_sequences(X_valseq, padding='post',maxlen=TT_max,dtype='float')
    Youtval=pad_sequences(Youtval, padding='post',maxlen=TT_max,dtype='float')
    X_trainseq=pad_sequences(X_trainseq, padding='post',maxlen=TT_max,dtype='float')
    Youttrain=pad_sequences(Youttrain, padding='post',maxlen=TT_max,dtype='float')
    #X_testseq=pad_sequences(X_testseq, padding='post',maxlen=TT_max)
    #Youttest=pad_sequences(Youttest, padding='post',maxlen=TT_max)

    print('..compiling model')
    mdninput_Lstm = keras.Input(shape=(None,window_size, inputDim))
    #mdninput_Lstm1=BatchNormalization(axis=-1)(mdninput_Lstm)
    TCNNOp1BN=TimeDistributed(Convolution1D(nb_filter=CNN1_filters, filter_length=CNN1_flength,activation=Relu_log,input_shape=(window_size, inputDim)))(mdninput_Lstm)
    TCNNOp1=BatchNormalization(axis=-1)(TCNNOp1BN)
    TCNNOp2=TimeDistributed(MaxPooling1D(window_size-CNN1_flength+1))(TCNNOp1)
#     TCNNOp=TimeDistributed(Reshape((CNN1_filters,1)))(TCNNOp2)
#     FCNNOp1=TimeDistributed(Convolution1D(nb_filter=CNN2_filters, filter_length=CNN2_flength, activation='relu'))    (TCNNOp)
#     #FCNNOp1=BatchNormalization(axis=-1)(FCNNOp1BN)
#     FCNNOp2=TimeDistributed(MaxPooling1D(3))(FCNNOp1)
    CNNOp=TimeDistributed(Flatten())(TCNNOp2)
    lstm_1=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(CNNOp)
    lstm_2a=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(lstm_1)
    lstm_2=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(lstm_2a)
    output=TimeDistributed(Dense(12, activation='linear'))(lstm_2)
    model = keras.models.Model(mdninput_Lstm,output)
    model.summary()
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, CNN1_filters, CNN1_flength))
    model.compile(optimizer='adam', loss='mse')


    #OutFileName='final_mfcc_'+Sub+'_F1_'+str(CNN1_filters)+'_len'+str(CNN1_flength)+'_F2_'+str(CNN2_filters)+'_len'+str(CNN2_flength)+'LSTMunits_'+str(NoUnits)+'_'
    OutFileName='JointALsub_Batch'+ str(BatchSize)+'_''_F1_'+str(CNN1_filters)+'_len'+str(CNN1_flength)+'_LSTMunits_'+str(NoUnits)+'_'
    fName=OutFileName
    print('..fitting model')

    checkpointer = ModelCheckpoint(filepath=OutDir+fName + '_.h5', verbose=0, save_best_only=True)
    checkpointer1 = ModelCheckpoint(filepath=OutDir+fName + '_weights.h5', verbose=0, save_best_only=True, save_weights_only=True)
    earlystopper =EarlyStopping(monitor='val_loss', patience=3)
    history=model.fit(X_trainseq[:,:,:,np.newaxis],Youttrain,validation_data=(X_valseq[:,:,:,np.newaxis],Youtval),nb_epoch=NoEpoch, batch_size=BatchSize,verbose=1,shuffle=True,callbacks=[checkpointer,checkpointer1,earlystopper])
    model.load_weights(OutDir+fName+ '_weights.h5')

    ValpredSeq=model.predict(X_valseq[:,:,:,np.newaxis]);
    #predSeq=model.predict(X_testseq[:,:,:,np.newaxis]);

    predSeq=np.empty((1,len(Youttest)), dtype=object);
    YtestOrg=np.empty((1,len(Youttest)), dtype=object);
    for i in np.arange(0,len(Youttest)):
        s_in=X_testseq[i]
        #s_in=s_in[np.newaxis,:,0:inputDim]
        val=model.predict(s_in);
        predSeq[0,i]=val
        #InSeq[0,i]=s_in
        YtestOrg[0,i]=Youttest[i]

    h=model.get_weights()
    scipy.io.savemat(OutDir+OutFileName+'out.mat', {'weights':h, 'OpPred':predSeq, 'OpOrg':YtestOrg,'ValOpPred':ValpredSeq, 'ValOpOrg': Youtval,'Time_Val': TT_Valid,'Time_Test':TT_Test })

