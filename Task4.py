
# coding: utf-8

# In[3]:

get_ipython().magic(u'matplotlib inline')
import time
import glob

import numpy as np
from skimage import data
from skimage import io
from skimage.transform import resize
from matplotlib.pyplot import imshow
from skimage.color import rgb2gray
from scipy import misc
from scipy import ndimage
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian
import matplotlib
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
import pandas as pd
import cv2
import datetime


# In[5]:

def get_im_cv2(path):
    img = cv2.imread(path)
    img = ndimage.gaussian_filter(img, sigma=3)
    resized = cv2.resize(img, (320, 320))
    return resized

def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..','sampleTrain', fld, '*.jpg')

        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            pic = rgb2gray(img)

            thresh = threshold_otsu(pic)
            binary = pic > thresh

            X_train.append(binary)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

X_train, y_train, X_train_id = load_train()

images_rs = np.asarray(X_train)
Y = np.asarray(y_train)
images_rs = images_rs.reshape([len(X_train), 320*320])


def load_test():
    path = os.path.join('..', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        pic = rgb2gray(img)
        
        thresh = threshold_otsu(pic)
        binary = pic > thresh
        X_test.append(binary)
        X_test_id.append(flbase)

    return X_test, X_test_id
X_test, X_test_id = load_test()

testX = np.asarray(X_test)
testX = testX.reshape([1000, 320*320])

clf = MLPClassifier(solver='sgd', alpha=0.0008, batch_size=100, learning_rate='adaptive', hidden_layer_sizes=(10, 2), random_state=1, early_stopping=True)
clf.fit(images_rs, Y)

predictions = clf.predict_proba(testX)

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)
create_submission(predictions, X_test_id, 'Task4')


# In[ ]:



