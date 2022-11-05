import tensorflow as tf
from tensorflow.kears import models
import random
import numpy as np


def GRU(maxlen, vector_dim, layers, dropout):
    """
    GRU
    """
    model = models.Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    model.add(sGRU(32))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model



def BGRU(maxlen, vector_dim, layers, dropout):
    """
    BRGU
    Arguments:
        (1)
    """
    model = models.Sequential()
    for i in range(layers):
        model.add(Bidirectional(CuDNNGRU(128, return_sequences=True),input_shape=(maxlen, vector_dim,)))
        model.add(Dropout(dropout))
        
    model.add(Bidirectional(CuDNNGRU(64)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy',getPrecision,getRecall,getF1])
    return model


def CNN(maxlen, vector_dim, layers, dropout):
    K.clear_session() 
    model = Sequential()
    #model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim, 1)))
    #channel,height,width
    model.add(Reshape((maxlen, vector_dim, 1), input_shape=(maxlen, vector_dim,)))
    model.add(Conv2D(256, (3,3),  padding = 'same', activation='relu'))
    #model.add(Conv2D(64, (3,3),  padding = 'same', activation='relu', input_shape=(maxlen, vector_dim, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()

    return model


def RNN(maxlen, vector_dim, layers, dropout):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    model.add(SimpleRNN(units=62))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model


def TextCNN(maxlen, vector_dim, layers, dropout):
    K.clear_session() 
    num_filters = 64
    kernel_size = [2, 4, 6, 8, 10]
    conv_action = 'relu'
    _input = Input(shape=(maxlen,vector_dim,))
    #_input = Reshape((maxlen, vector_dim, 1), input_shape=(maxlen, vector_dim,))
    #_embed = Embedding(304, 256, input_length=maxlen)(_input)
    #_embed = SpatialDropout1D(0.15)(_embed)
    #_embed = SpatialDropout1D(0.15)(_input)
    warppers = []
    for _kernel_size in kernel_size:
        conv1d = Conv1D(filters=256, kernel_size=_kernel_size, activation=conv_action, padding="same")(_input)
        #conv1d = Conv1D(filters=32, kernel_size=_kernel_size, activation=conv_action, padding="same")(_embed)
        warppers.append(MaxPool1D(48)(conv1d))

    fc = concatenate(warppers)
    fc = Flatten()(fc)
    fc = Dropout(0.2)(fc)
    # fc = BatchNormalization()(fc)
    fc = Dense(128, activation='relu')(fc)
    fc = Dropout(0.2)(fc)
    # fc = BatchNormalization()(fc)
    preds = Dense(1, activation='sigmoid')(fc)
    model = Model(inputs=_input, outputs=preds)
    #model.compile(loss='categorical_crossentropy',
    #              optimizer='adam',
    #              metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',getPrecision,getRecall,getF1])
    return model


def BLSTM(maxlen, vector_dim, layers, dropout):
    model = Sequential()
    #model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    #model.add(Bidirectional(sLSTM(64, activation='tanh',)))
    model.add(Bidirectional(CuDNNLSTM(64, ), input_shape=(maxlen, vector_dim,)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model


def LSTM(maxlen, vector_dim, layers, dropout):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    #model.add(Reshape((maxlen, vector_dim,), input_shape=(maxlen, vector_dim,)))
    #model.add(CuDNNLSTM(64, return_sequences=True,input_shape=(maxlen, vector_dim,)))
    #model.add(CuDNNLSTM(64, return_sequences=True, input_shape=(maxlen, vector_dim,)))
    model.add(sLSTM(64, activation='tanh', return_sequences=True))
    model.add(Dropout(dropout))
    #model.add(CuDNNLSTM(64))
    model.add(sLSTM(64, activation='tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model


def CNN_LSTM(maxlen, vector_dim, layers, dropout):
    model = Sequential()
    #model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    model.add(Reshape((maxlen, vector_dim, 1), input_shape=(maxlen, vector_dim,)))
    #model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 2, 1)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(sLSTM(64, activation='tanh'))
    #model.add(CuDNNLSTM(64))
    model.add(Dropout(dropout))
    model.add(Dense(32))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model

