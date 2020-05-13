import sys

if len(sys.argv) < 2:
    sys.stderr.write('please provide a folder that contain test and train dataset \n')
    sys.exit(0)
    
DATA_DIR = sys.argv[1]


import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
import tensorflow as tf 

#TRAIN_DIR = 'E:/dataset/CSCE990/ train'
#TEST_DIR = 'E:/dataset/CSCE990/ test'
IMG_SIZE = 49152   # Each constellation is considered as an image


MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic')

train_data = read.cvs(DATA_DIR+'/train') 
test_data = read.cvs(DATA_DIR+'/test')


def compute_cumulant4(data):     # Extracting 4th-order cumulnats
    channels = data.shape[0]
    tensor = np.zeros((channels, channels, channels, channels))
    E = np.zeros((channels, channels)) + np.inf
    for ch0 in range(channels):
        for ch1 in range(ch0, channels):
            if E[ch0, ch1]==np.inf:
                E[ch0, ch1] = np.mean(data[ch0, :] * data[ch1, :])
                E[ch1, ch0] = E[ch0, ch1]
            for ch2 in range(ch1, channels):
                if E[ch0, ch2]==np.inf:
                    E[ch0, ch2] = np.mean(data[ch0, :] * data[ch2, :])
                    E[ch2, ch0] = E[ch0, ch2]
                if E[ch1, ch2]==np.inf:
                    E[ch1, ch2] = np.mean(data[ch1, :] * data[ch2, :])
                    E[ch2, ch1] = E[ch1, ch2]
                for ch3 in range(ch2, channels):
                    if E[ch0, ch3]==np.inf:
                        E[ch0, ch3] = np.mean(data[ch0, :] * data[ch3, :])
                        E[ch3, ch0] = E[ch0, ch3]
                    if E[ch1, ch3]==np.inf:
                        E[ch1, ch3] = np.mean(data[ch1, :] * data[ch3, :])
                        E[ch3, ch1] = E[ch1, ch3]
                    if E[ch2, ch3]==np.inf:
                        E[ch2, ch3] = np.mean(data[ch2, :] * data[ch3, :])
                        E[ch3, ch2] = E[ch2, ch3]
                    cumulant = np.mean(data[ch0, :] * data[ch1, :] * data[ch2, :] * data[ch3, :]) - E[ch0, ch1] * E[ch2, ch3] - E[ch0, ch2] * E[ch1, ch3] - E[ch0, ch3] * E[ch1, ch2]

                    tensor[ch0, ch1, ch2, ch3] = cumulant
                    tensor[ch0, ch1, ch3, ch2] = cumulant
                    tensor[ch0, ch2, ch1, ch3] = cumulant
                    tensor[ch0, ch2, ch3, ch1] = cumulant
                    tensor[ch0, ch3, ch1, ch2] = cumulant
                    tensor[ch0, ch3, ch2, ch1] = cumulant
                    
                    tensor[ch1, ch0, ch2, ch3] = cumulant
                    tensor[ch1, ch0, ch3, ch2] = cumulant
                    tensor[ch1, ch2, ch0, ch3] = cumulant
                    tensor[ch1, ch2, ch3, ch0] = cumulant
                    tensor[ch1, ch3, ch0, ch2] = cumulant
                    tensor[ch1, ch3, ch2, ch0] = cumulant
                    
                    tensor[ch2, ch1, ch0, ch3] = cumulant
                    tensor[ch2, ch1, ch3, ch0] = cumulant
                    tensor[ch2, ch0, ch1, ch3] = cumulant
                    tensor[ch2, ch0, ch3, ch1] = cumulant
                    tensor[ch2, ch3, ch1, ch0] = cumulant
                    tensor[ch2, ch3, ch0, ch1] = cumulant
                    
                    tensor[ch3, ch1, ch2, ch0] = cumulant
                    tensor[ch3, ch1, ch0, ch2] = cumulant
                    tensor[ch3, ch2, ch1, ch0] = cumulant
                    tensor[ch3, ch2, ch0, ch1] = cumulant
                    tensor[ch3, ch0, ch1, ch2] = cumulant
                    tensor[ch3, ch0, ch2, ch1] = cumulant
    return tensor



for i in np.size(train_data, 0)
	cumulant[i] = compute_cumulant4(train_data[i,:])



for i in np.size(train_data, 0)
	train_data.append([i])


# training Process

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(12, 3),activation='relu',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(AvePooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(AvePooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(AvePooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))

tf.reset_default_graph() 
convnet = input_data(shape =[None, 32440,1025, 1], name ='train_data') 
  
convnet = conv_2d(convnet, 12, 3, activation ='relu') 
convnet = Ave_pool_2d(1, 2) 
...  
convnet = conv_2d(convnet, 24, 3, activation ='relu') 
convnet = Ave_pool_2d(1, 2) 
...
convnet = conv_2d(convnet, 32, 3, activation ='relu') 
convnet = Ave_pool_2d(1, 2) 
...   
model = tflearn.DNN(convnet, tensorboard_dir ='log') 



train = train_data[:-500] 
test = train_data[-500:] 
  
'''Setting up the features and lables'''
# X-Features & Y-Labels 
  
X = np.array([i[0] for i in train]).reshape(-1, 32440,1025, 1) 
Y = [i[1] for i in train] 
test_x = np.array([i[0] for i in test]).reshape(-1, 32440,1025, 1) 
test_y = [i[1] for i in test] 
  
'''Fitting the data into our model'''
# epoch = 5 taken 
model.fit({'input': X}, {'targets': Y}, n_epoch = 5,  
    validation_set =({'input': test_x}, {'targets': test_y}),  
    snapshot_step = 500, show_metric = True, run_id = MODEL_NAME) 
model.save(MODEL_NAME)

for num, data in enumerate(test_data[:,:]): 

      
    img_num = data[1] 
    img_data = data[0] 
      
    y = fig.add_subplot(4, 5, num + 1) 
    orig = img_data 
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1) 
  
    fashion_model.add(Dense(num_classes, activation='relu'))
    fashion_model.add(softmax())

    model_out = model.predict([test_data])[0] 
      
    if np.argmax(model_out) == 1: str_label ='BPSK'
    if np.argmax(model_out) == 2: str_label ='QPSK'
    if np.argmax(model_out) == 3: str_label ='128QAM'
    if np.argmax(model_out) == 4: str_label ='256QAM'
plt.show() 
