import tensorflow as tf
from keras import backend as K
from keras.preprocessing import sequence
from keras.utils import pad_sequences
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.models import *
from keras.layers.core import Masking, Dense, Dropout, Activation
from keras.layers.core import *
#from keras.layers.recurrent import LSTM,GRU
from keras.layers import CuDNNLSTM, CuDNNGRU, PReLU, Reshape,Conv2D, MaxPooling2D, Flatten,SimpleRNN,Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding,SpatialDropout1D, Input, TimeDistributed,MaxPooling1D, GlobalAveragePooling1D
from keras.layers import LSTM as sLSTM
from keras.layers import GRU as sGRU
from keras.layers import Multiply

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
#from preprocess_dl_Input_version5 import *
from keras.layers import Bidirectional
from collections import Counter
import numpy as np
import pickle, joblib
import random
import time
import math
import os

from keras import metrics

#tf.compat.v1.keras.layers.CuDNNGRU

RANDOMSEED = 2020  # for reproducibility
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def getRecall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def getPrecision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def getF1(y_true, y_pred):
    """F1-score"""
    precision = getPrecision(y_true, y_pred)
    recall = getRecall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = 200

    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    #if SINGLE_ATTENTION_VECTOR:
    #    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul




def BGRU(maxlen, vector_dim, layers, dropout):
    print('Build model...')
    model = Sequential()
    #model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    
    for i in range(layers):
        #model.add(Bidirectional(GRU(units=256, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True)))
        model.add(Bidirectional(CuDNNGRU(128, return_sequences=True),input_shape=(maxlen, vector_dim,)))
        #model.add(CuDNNGRU(units=256, return_sequences=True, name='GRU'+str(i)))
        model.add(Dropout(dropout))
        
    #model.add(Bidirectional(GRU(units=64, activation='tanh', recurrent_activation='hard_sigmoid')))
    model.add(Bidirectional(CuDNNGRU(64)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['TP_count', 'FP_count', 'FN_count', 'precision', 'recall', 'fbeta_score'])
    # model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model



def GRU(maxlen, vector_dim, layers, dropout):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    #model.add(Embedding(3800, 32, input_shape=380))
    #model.add(Dropout(dropout))
    model.add(sGRU(32))
    #model.add(CuDNNGRU(32, input_shape=(maxlen, vector_dim,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
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


#from Attention import Self_Attention
def attention_model(maxlen, vector_dim, layers, dropout):
    x_input = Input(shape=(maxlen, vector_dim))
    O_seq = Self_Attention(64)(x_input)
    #O_seq = Conv1D(128, 3, padding='valid',
    #                   activation='relu', strides=1)(O_seq)
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(dropout)(O_seq)
    outputs = Dense(1, activation='sigmoid')(O_seq)
    model = Model(inputs=x_input, outputs=outputs)
    #adam = Adam(lr=CNNConfig.learn_rate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model


def attention_lstm(maxlen, vector_dim, layers, dropout):
    K.clear_session() 
    inputs = Input(shape=(maxlen, vector_dim,))
    attention_mul = attention_3d_block(inputs)
    attention_mul = sLSTM(64, activation='tanh', return_sequences=True)(attention_mul)
    attention_mul = Dropout(dropout)(attention_mul)
    attention_mul = sLSTM(64, activation='tanh')(attention_mul)
    attention_mul = Dropout(dropout)(attention_mul)
    
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model


def lstm_attention(maxlen, vector_dim, layers, dropout):
    K.clear_session() 
    inputs = Input(shape=(maxlen, vector_dim,))
    lstm_out = sLSTM(64, activation='tanh', return_sequences=True)(inputs)
    lstm_out = Dropout(dropout)(lstm_out)
    lstm_out = sLSTM(64, activation='tanh', return_sequences=True)(lstm_out)
    lstm_out = Dropout(dropout)(lstm_out)
    
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',getPrecision,getRecall,getF1])
    model.summary()
    return model



def process_sequences_shape(sequences, maxLen, vector_dim):
    samples = len(sequences)
    #nb_samples = np.zeros((samples, maxLen, vector_dim), 'float16')
    nb_samples = np.zeros((samples, maxLen, vector_dim))

    i = 0
    for sequence in sequences:
        m = 0
        for vectors in sequence:
            n = 0
            for values in vectors:
                nb_samples[i][m][n] += values
                n += 1
            m += 1
            if m == maxLen:
                break
        i += 1
    return nb_samples


def generator_of_data(data, labels, batchsize, maxlen, vector_dim):
    iter_num = int(len(data) / batchsize)
    i = 0
    
    while iter_num:
        batchdata = data[i:i + batchsize]
        batched_input = process_sequences_shape(batchdata, maxLen=maxlen, vector_dim=vector_dim)
        #batched_input = batchdata
        batched_labels = labels[i:i + batchsize]
        yield (batched_input, batched_labels)
        i = i + batchsize
        
        iter_num -= 1
        if iter_num == 0:
            iter_num = int(len(data) / batchsize)
            i = 0


def main(traindataSet_path, testdataSet_path, realtestpath, weightpath, resultpath, batch_size, maxlen, vector_dim, layers, dropout):
    print("Loading data...")
    if not os.path.exists('model'):
        os.makedirs('model')
    #model = BGRU(maxlen, vector_dim, layers, dropout)
    #model = BLSTM(maxlen, vector_dim, layers, dropout)
    #model = LSTM(maxlen, vector_dim, layers, dropout)
    #model = GRU(maxlen, vector_dim, layers, dropout)
    #model = CNN(maxlen, vector_dim, layers, dropout)
    #model = CNN_LSTM(maxlen, vector_dim, layers, dropout)
    #model = TextCNN(maxlen, vector_dim, layers, dropout)
    #model = RNN(maxlen, vector_dim, layers, dropout)
    
    #model = attention_model(maxlen, vector_dim, layers, dropout)
    #model = attention_lstm(maxlen, vector_dim, layers, dropout)
    model = lstm_attention(maxlen, vector_dim, layers, dropout)
    
    
    #model.load_weights(weightpath)  #load weights of trained model
    #model.load_weights('./model/1kda3k_blstm_shuffle_10000_32vc')  #load weights of trained model
    
    #with open('datavec/vec_reentrancy_train_token_data.pkl','rb') as f:
    #with open('datavec/vec_normalize_reentrancy_train_token_data.pkl','rb') as f:
    #with open('datavec/vec_dataflow_normalize_reentrancy_train_token_data.pkl','rb') as f:
    with open('datavec/vec_dfcf_normalize_reentrancy_train_token_data.pkl','rb') as f:
    #with open('datavec/vec_s2v_dataflow_normalize_reentrancy_train_token_data.pkl','rb') as f:

    #with open('datavec/vec_normalize_timestamp_train_token_data.pkl','rb') as f:
    #with open('datavec/vec_dfcf_normalize_timestamp_train_token_data.pkl','rb') as f:

        data = joblib.load(f)
    with open('datavec/vec_reentrancy_valid_token_data.pkl','rb') as f:
        testdata = pickle.load(f)

    labels = []
    dataset = []
    filenames = []
    # no augmentation and double
    dataset += data['good']['vectors']
    dataset += data['bad']['vectors']
    labels += [0 for i in range(len(data['good']['vectors']))]
    labels += [1 for i in range(len(data['bad']['vectors']))]
    labels = np.array(labels)

    testlabels = []
    testdataset = []
    testdataset += testdata['good']['vectors']
    testdataset += testdata['bad']['vectors']
    testlabels += [0 for i in range(len(testdata['good']['vectors']))]
    testlabels += [1 for i in range(len(testdata['bad']['vectors']))]

    np.random.seed(RANDOMSEED)
    np.random.shuffle(dataset)
    np.random.seed(RANDOMSEED)
    np.random.shuffle(labels)
    np.random.seed(RANDOMSEED)
    np.random.shuffle(testdataset)
    np.random.seed(RANDOMSEED)
    np.random.shuffle(testlabels)

    
    print('Vectorize...')
    #X_test = process_sequences_shape(X_test, maxLen=maxlen, vector_dim=vector_dim)
    #print(len(testdataset))
    X_train = process_sequences_shape(dataset, maxLen=maxlen, vector_dim=vector_dim)
    y_train = labels
    

    #from sklearn.model_selection import StratifiedKFold
    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOMSEED)
    #X_test = process_sequences_shape(testdataset, maxLen=maxlen, vector_dim=vector_dim)
    #y_test = testlabels
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOMSEED)
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.9, random_state=13433)
    #np.random.seed(RANDOMSEED)
    #np.random.shuffle(X_train)
    #np.random.seed(RANDOMSEED)
    #np.random.shuffle(y_train)

    print('length :::: ' + str(len(y_train)))
    train_generator = generator_of_data(X_train, y_train, batch_size, maxlen, vector_dim)
    all_train_samples = len(X_train)
    steps_epoch = int(all_train_samples / batch_size)

    print("Train...")
    checkpoint = ModelCheckpoint('./model/temp', monitor='val_getF1', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
    #checkpoint = ModelCheckpoint('./model/best_aug_normalize_textcnn', monitor='val_getF1', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
    #checkpoint = ModelCheckpoint('./model/best_normalize_blstm', monitor='val_getF1', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
    #tbCallBack = TensorBoard(log_dir="./tsboard", histogram_freq=1,write_grads=True)
    t1 = time.time()
    #history = model.fit_generator(train_generator, validation_data=(X_test,y_test), workers=8, pickle_safe=True, steps_per_epoch=steps_epoch, epochs=100, callbacks=[checkpoint])
    model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=200, batch_size=batch_size, callbacks=[checkpoint])
    t2 = time.time()
    train_time = t2 - t1

    #loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    #print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    #plot_history(history)

    #model.save_weights('./model/augx5_normalize_3-layer-BGRU')
    #model.save_weights('./model/textcnn')
    #model.save_weights('./model/blstm')
    #with open('history/aug_normalize_3-layer-BGRU','wb') as f:
    #with open('history/textcnn','wb') as f:
    #with open('history/blstm','wb') as f:
    #    pickle.dump(history.history, f)
    print("Train Done...", train_time)



if __name__ == "__main__":
    batchSize = 32
    vectorDim = 64
    maxLen = 200
    layers = 2
    dropout = 0.2
    traindataSetPath = "./dl_input_shuffle/cdg_ddg/train/"
    testdataSetPath = "./dl_input_shuffle/cdg_ddg/test/"
    realtestdataSetPath = "data/"
    #weightPath = './model/BRGU'
    weightPath = './model/augx5_normalize_3-layer-BGRU'
    resultPath = "./result/BGRU/BGRU"
    main(traindataSetPath, testdataSetPath, realtestdataSetPath, weightPath, resultPath, batchSize, maxLen, vectorDim, layers, dropout)



