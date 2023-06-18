# Plant-Seedlings-Classification
Determine the species of a seedling from an image

link of competition : https://www.kaggle.com/c/plant-seedlings-classification

Data: https://www.kaggle.com/c/plant-seedlings-classification/data

Author: https://www.kaggle.com/miklgr500

general steps:
  - assign class for each type by number from 0 to 12
  - Build CNN by using written functions:
  
                   -get_model(..) -----in this function we build CNN (convolution layers +Nural network)
                         -conv_layer(..)  -----------is used to build convolution layers,#aftr 2 convolution layer 
                                                   ,we apply maxPooling to reduce the size of  data ,reduce overfitting ,.....   
                          -we apply Flatten() on the o/p of convolution layers
                          -dense_set(..)-------- we use it to build hidden layer and outpt layer in ANN
                          
   - Traning our model using train_model(),we train it on GTX 950 and load the weights in file "model_weight_Adam.hdf5"
                       -then we load these weights to our model  (note:we train our model using 2 optimizers (Adam ,SGD))
                       
   - Test our model  using test_model(..) 
   
   
   Notes:-
           - we apply some transformations on our train images to increas the size of data using ImageDataGenerator( ....)
           - we use some functions to get plant class on image,.....
  

