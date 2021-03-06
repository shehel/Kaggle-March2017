#########################
#The Nature Conservancy Fisheries Monitoring - Kaggle Competition - March 2017
#https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring
#Goal: Create a model to predict the species of fish present in a given image captured in fishing boats. 
#      This is so that actions can be taken against fishermen who catch endangered species of fishes 
#
#Problem is complicated due to the background noise in the images such as people, other fishes, boat, lighting etc
#
#This is the best performing model I developed in the 2 weeks I worked on this problem which was in the top 10% based on
#performance. It uses OpenCV, SciKit-learn and Keras for data processing and creating CNN models respectively. 
#
#The performance I could potentially achieve was also greatly limited by the processing power and GPU I had at my
#disposal. Model with larger convolutuion layers and greater image size and number of images would also lead to far 
#better results. 
##########################



# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

from keras.layers.normalization import BatchNormalization # batch normalisation
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras.regularizers import l2 # L2-regularisation
from sklearn.cross_validation import KFold
import numpy as np
import os
import glob
import cv2
import datetime
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras import backend as K

import warnings
np.random.seed(123)



# In[ ]:

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (128, 128))
    return resized

#Load training data
def load_train():
    X_train = []
    X_train_id = []
    y_train = []

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..','sampleTrain', fld, '*.jpg')
        files = glob.glob(path)
        count = 0
        for fl in files:
            #if count > 10:
             #   break
            count = count+1
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    #print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id





def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

#Pre-processing loaded images (Both test and train data)
def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

train_data, train_target, train_id = read_and_normalize_train_data()

#Deep layered network
def create_model():
    model = Sequential()
    
    l2_lambda = 0.0001 # use 0.0001 as a L2-regularisation factor
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 128, 128), dim_ordering='th'))
    model.add(BatchNormalization(axis = 1))

    model.add(Convolution2D(4, 2, 2, activation='relu', W_regularizer=l2(l2_lambda), dim_ordering='th', init= 'he_normal') )
    model.add(BatchNormalization(axis = 1))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 2, 2, activation='relu', W_regularizer=l2(l2_lambda), dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 2, 2, activation='relu', W_regularizer=l2(l2_lambda),dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 2, 2, activation='relu', W_regularizer=l2(l2_lambda), dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 2, 2, activation='relu', W_regularizer=l2(l2_lambda),dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 2, 2, activation='relu', W_regularizer=l2(l2_lambda), dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1)) 
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 2, 2, activation='relu', W_regularizer=l2(l2_lambda), dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))  
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(32, 2, 2, activation='relu',  W_regularizer=l2(l2_lambda),dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(32, 2, 2, activation='relu', W_regularizer=l2(l2_lambda),dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(32, 2, 2, activation='relu', W_regularizer=l2(l2_lambda),dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 2, 3, activation='relu',  W_regularizer=l2(l2_lambda),dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 2, 3, activation='relu', W_regularizer=l2(l2_lambda),dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 2, 3, activation='relu', W_regularizer=l2(l2_lambda),dim_ordering='th', init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_regularizer=l2(l2_lambda), init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))

    
    model.add(Dense(512, activation='relu',  W_regularizer=l2(l2_lambda), init= 'he_normal'))
    model.add(BatchNormalization(axis = 1))

    model.add(Dense(8,  W_regularizer=l2(l2_lambda), activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model
 
 #Create the defined model and run CV
 def run_cross_validation_create_models(train_data, train_target, nfolds=10):
    # input image dimensions
    batch_size = 64
    nb_epoch = 100
    random_state = 51

    yfull_train = dict()
    kf = KFold(len(train_target), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models

def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret

#Pseudo-ensemble model by averaging results of models 
#with different training data
def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

num_folds = 5

#Method that uses the created models to predict kind of fish and create a submission 
#file 
def run_cross_validation_process_test(info_string, models):
    batch_size = 64
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)
    test_data, test_id = read_and_normalize_test_data()

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'task5' + info_string                 + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


    
info_string, models = run_cross_validation_create_models(train_data, train_target, num_folds)

run_cross_validation_process_test(info_string, models)
    


# In[ ]:



