# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:47:20 2018

@author: Lenovo
"""

import pandas as pd
import numpy as np
import os
import imageio

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Maximum
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize as imresize
from tqdm import tqdm

"""from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))"""


BATCH_SIZE = 16
EPOCHS = 30
RANDOM_STATE = 11

CLASS = {
    'Black-grass': 0,
    'Charlock': 1,
    'Cleavers': 2,
    'Common Chickweed': 3,
    'Common wheat': 4,
    'Fat Hen': 5,
    'Loose Silky-bent': 6,
    'Maize': 7,
    'Scentless Mayweed': 8,
    'Shepherds Purse': 9,
    'Small-flowered Cranesbill': 10,
    'Sugar beet': 11
}

INV_CLASS = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet'
}

# Dense layers set   
#it is deffierent types of keras layers  which 
# *1st layer is Dropout layer is used to rprevent the overvitting
#*2nd layer is Dense layer (fully connected layer) i which we determine the number of nerons ,no activation is applied (ie. "linear" activation: a(x) = x).
#*3rd layer is BatchNormalization ,it Normalizes the activations of the previous layer at each batch
#*4th layer is Activation Applies an activation function to an output.
#and these layers connected by the traditional way of MODEL
#this function returns the last layer to be as link to the rest of layers of the model that we will build
def dense_set(inp_layer, n, activation, drop_rate=0.):   #inp_layer ---input layer , n---number of neurons in Dense layer , activation---name of activation function
    dp = Dropout(drop_rate)(inp_layer)                                        
    dns = Dense(n)(dp)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation=activation)(bn)
    return act

# Conv. layers set
#1st layer is normal convolution layer ,we determine to add  zeroPadding2D layer before it according to  value of zp_flag
#2nd layer is BatchNormalization
#3rd layer is Activation and use   LeakyReLU 
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3),strides=(1,1), zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1,1))(feature_batch)                  # Zero-padding layer for 2D input (e.g. picture)   This layer can add rows and columns of zeros
																										# at the top, bottom, left and right side of an image tensor
    else:                                                                                            #If tuple of 2 ints:interpreted as two different symmetric padding values for height and width      
        zp = feature_batch                                                               #`(symmetric_height_pad, symmetric_width_pad)`.
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    act = LeakyReLU(1/10)(bn)
    return act

##############simple model  ##############################   
 #build the model of type Model
#creating this model  is done as following:  name_model=Model(inputs=input_layer, outputs=output_layer)
#and This model will include all layers required in the computation of output_layer  given input_layer .
#each layer(except the 1st layer "input layer") in this model build as following :  layer_name=layer_name_that_used(...................) (name_of_previous layer)
def get_model():  ########model is CNN that consists of Convolution layer/s and Artifial Neural network (ANN)
    inp_img = Input(shape=(51, 51, 3))  #1st layer in CNN 

    # 51
    conv1 = conv_layer(inp_img, 64, zp_flag=False)
    conv2 = conv_layer(conv1, 64, zp_flag=False)   
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)             #aftr 2 convolution layer ,we apply maxPooling to reduce the size of  data ,reduce overfitting ,.........         
    # 23
    conv3 = conv_layer(mp1, 128, zp_flag=False)
    conv4 = conv_layer(conv3, 128, zp_flag=False)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
    # 9
    conv7 = conv_layer(mp2, 256, zp_flag=False)
    conv8 = conv_layer(conv7, 256, zp_flag=False)
    conv9 = conv_layer(conv8, 256, zp_flag=False)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv9)
    # 1                                     ################# end  of conolution layers###############
    # dense layers                                   
    flt = Flatten()(mp3)                                 #flaten the o/p of convolution layer into single vector ,to input it to ANN           
    ds1 = dense_set(flt, 128, activation='tanh')           #hidden layer made from dense_set we illustrated above
    out = dense_set(ds1, 12, activation='softmax')    #output layer of 12 neurons one for each class of seddings
	
   ####################Build Model model#############################
    model = Model(inputs=inp_img, outputs=out)           
          #ptimzers
    # The first 50 epochs are used by Adam opt.
    # Then 30 epochs are used by SGD opt.
    #mypotim = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  #Adam optimizer
    mypotim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)   #stochastic gradient desent optimizer
	             #combile model to determine some parameters
    model.compile(loss='categorical_crossentropy',
                 				  optimizer=mypotim,
                  				 metrics=['accuracy'])
	########################################################################
    model.summary()   ##return brief inforation about construction of Or NN
    return model


def get_callbacks(filepath, patience=5):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1)
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [lr_reduce, msave]
###########################Train model######################################
# I trained model about 12h on GTX 950.
def train_model(img, target):
    callbacks = get_callbacks(filepath='model_weight_SGD.hdf5', patience=6)
    gmodel = get_model()      #model constuction
    gmodel.load_weights(filepath='model_weight_Adam.hdf5')    #load the weights to our model,these weights are obtained from the traning our model on GTX 950
    x_train, x_valid, y_train, y_valid = train_test_split(
                                                        img,
                                                        target,
                                                        shuffle=True,
                                                        train_size=0.8,
                                                        random_state=RANDOM_STATE
                                                        )
    gen = ImageDataGenerator(                            #data Augmenatation make some transformations on the existing data,to increase  our data that used in train / test...
            rotation_range=360.,                                 # we make object "gen" that contais some transformation and we will applied on train data
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True
    )                                                                #fit data using fit_generator(.......)                           
    gmodel.fit_generator(gen.flow(x_train, y_train,batch_size=BATCH_SIZE),      #gen.flow(.........) is train data after we applied some transformations using  gen
               steps_per_epoch=10*len(x_train)/BATCH_SIZE,
               epochs=EPOCHS,
               verbose=1,
               shuffle=True,
               validation_data=(x_valid, y_valid),
               callbacks=callbacks)

def test_model(img, label):
    gmodel = get_model()
    gmodel.load_weights(filepath='../input/plant-weight/model_weight_SGD.hdf5')
    prob = gmodel.predict(img, verbose=1)
    pred = prob.argmax(axis=-1)
    sub = pd.DataFrame({"file": label,
                         "species": [INV_CLASS[p] for p in pred]})
    sub.to_csv("sub.csv", index=False, header=True)

# Resize all image to 51x51 
def img_reshape(img):
    img = imresize(img, (51, 51, 3))
    return img

# get image tag
def img_label(path):
    return str(str(path.split('/')[-1]))

# get plant class on image
def img_class(path):
    return str(path.split('/')[-2])

# fill train and test dict
def fill_dict(paths, some_dict):
    text = ''
    if 'train' in paths[0]:
        text = 'Start fill train_dict'
    elif 'test' in paths[0]:
        text = 'Start fill test_dict'

    for p in tqdm(paths, ascii=True, ncols=85, desc=text):
        img = imageio.imread(p)
        img = img_reshape(img)
        some_dict['image'].append(img)
        some_dict['label'].append(img_label(p))
        if 'train' in paths[0]:
            some_dict['class'].append(img_class(p))

    return some_dict

# read image from dir. and fill train and test dict
def reader():
    file_ext = []
    train_path = []
    test_path = []

    for root, dirs, files in os.walk('../input'):
        if dirs != []:
            print('Root:\n'+str(root))
            print('Dirs:\n'+str(dirs))
        else:
            for f in files:
                ext = os.path.splitext(str(f))[1][1:]

                if ext not in file_ext:
                    file_ext.append(ext)

                if 'train' in root:
                    path = os.path.join(root, f)
                    train_path.append(path)
                elif 'test' in root:
                    path = os.path.join(root, f)
                    test_path.append(path)
    train_dict = {
        'image': [],
        'label': [],
        'class': []
    }
    test_dict = {
        'image': [],
        'label': []
    }

    #train_dict = fill_dict(train_path, train_dict)
    test_dict = fill_dict(test_path, test_dict)
    return train_dict, test_dict
# I commented out some of the code for learning the model.
def main():
    train_dict, test_dict = reader()
    #X_train = np.array(train_dict['image'])
    #y_train = to_categorical(np.array([CLASS[l] for l in train_dict['class']]))

    X_test = np.array(test_dict['image'])
    label = test_dict['label']
    
    # I do not recommend trying to train the model on a kaggle.
    #train_model(X_train, y_train)
    test_model(X_test, label)

if __name__=='__main__':
    main()
