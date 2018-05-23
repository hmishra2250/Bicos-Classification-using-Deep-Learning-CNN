import os
import numpy as np
import cv2
from keras.utils import to_categorical


files = ['jpg/' + x  for x in os.listdir('jpg/')]
labels = np.loadtxt('labels.csv', dtype='int32', delimiter=',')
filelabel = np.array([ labels[int((x.split('_')[-1]).split('.')[0]) - 1] for x in files])
cat_labels = to_categorical(filelabel)
np.save('cat_labels.npy', cat_labels)
np.save('filelabel.npy', filelabel)

data = []
for file in files:
    data.append(cv2.resize(cv2.imread(file, 1), (128, 128)))

data = np.array(data)
np.save('data128.npy', data)


data = []
for file in files:
    data.append(cv2.resize(cv2.imread(file, 1), (64, 64)))

data = np.array(data)
np.save('data64.npy', data)

data = []
for file in files:
    data.append(cv2.resize(cv2.imread(file, 1), (224, 224)))

data = np.array(data)
np.save('data224.npy', data)
