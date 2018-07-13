# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:55:25 2018

@author: Weiyu_Lee
"""

import os
os.environ['KERAS_BACKEND']='tensorflow'

from keras.layers import Conv2D
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint,TensorBoard

import numpy as np
import os
import pickle

def load_pickle_data(path):
    
    f = open(os.path.join(path, "normal_data.pkl"), "rb")
    n_image = np.array(pickle.load(f))
    n_code = np.array(pickle.load(f))
    f.close()
            
    f = open(os.path.join(path, "abnormal_data.pkl"), "rb")
    ab_image = np.array(pickle.load(f))
    ab_code = np.array(pickle.load(f))
    f.close()        
    
    return n_image, n_code, ab_image, ab_code

epochs = 100
batch_size = 16

#vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 
#
#add_model = Sequential()
#add_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
#add_model.add(Dense(8192, activation='relu'))
#add_model.add(Dense(4096, activation='relu'))
#add_model.add(Dense(4096, activation='linear'))
#
#model = Model(inputs=vgg16_model.input, outputs=add_model(vgg16_model.output))
#
#for layer in vgg16_model.layers:
#    layer.trainable = False
    
input_shape = (224, 224, 3)

model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(8912, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(4096, activation='linear')
])

model.summary()


model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=1e-4, beta_1=0.5, beta_2=0.9, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

# checkpoint
filepath="/home/sdc1/model/MDetection/detection_model/vgg16_model-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

# tensorboard
tensorboard = TensorBoard(log_dir='/home/sdc1/model/MDetection/detection_model/log/', histogram_freq=0, write_graph=True, write_images=True)

callbacks_list = [checkpoint, tensorboard]

print("Loading data...")
n_image, n_code, ab_image, ab_code = load_pickle_data("/home/sdc1/dataset/ICPR2012/training_data/scanner_A/classfied_data/224x224/")

train_data = np.concatenate((n_image, ab_image))
label_data = np.concatenate((n_code, ab_code))

print("Train data: [{}]".format(train_data.shape))
print("Label data: [{}]".format(label_data.shape))

model.fit(train_data, label_data, 
          epochs=epochs, 
          batch_size=batch_size, 
          validation_split=0.2,
          shuffle=True,
          callbacks=callbacks_list)
