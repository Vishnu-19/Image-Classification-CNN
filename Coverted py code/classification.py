# -*- coding: utf-8 -*-
"""Classification.ipynb

# Image Classification on CIFAR-10 dataset

###The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

###There are 50000 training images and 10000 test images.
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

"""## Data Loading

"""

from keras.datasets import cifar10
(xtrain,ytrain),(xtest,ytest) = cifar10.load_data()

"""## Data Visualization"""

i = 1005
plt.imshow(xtrain[i])
print(ytrain[i])

#Plotting images to visualize
w_grid = 15
l_grid = 15
fig,axes = plt.subplots(l_grid,w_grid,figsize = (25,25))
axes = axes.ravel();

for i in np.arange(0,l_grid*w_grid):
    index = np.random.randint(0,50000)
    axes[i].imshow(xtrain[index])
    axes[i].set_title(ytrain[index])
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)

"""## Data Preprocessing"""

#Converting data to 32 bit floats
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')

#Normalize
xtrain = xtrain/255
xtest = xtest/255

ytrain = keras.utils.to_categorical(ytrain,10)
ytest = keras.utils.to_categorical(ytest,10)

Input_shape = (32,32,3)

"""## Model Training"""

# Import
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

"""## Model 1

"""

# Creating the model
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (3,3), activation='relu',padding="same",input_shape=Input_shape))
model.add(Conv2D(filters = 16, kernel_size = (3,3), activation='relu',padding="same"))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation='relu',padding="same"))
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation='relu',padding="same"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(units =400,activation='relu'))
model.add(Dense(units = 200,activation='relu'))

model.add(Dense(units = 10,activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

hist = model.fit(xtrain,ytrain,batch_size= 32 ,epochs = 50,verbose=1,shuffle = True,validation_data=(xtest,ytest))

best=max(hist.history['val_accuracy']) *100
print('Best Accuracy = '+str(round(best,2)))

score = model.evaluate(xtest,ytest)
model.save('model.h5')

#Plotting the data
epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,hist.history['accuracy'],label='Train Accuracy')
plt.plot(epochs,hist.history['val_accuracy'],label='Val Accuracy')
plt.title('CIFAR-Model-1')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.grid()
plt.legend()
plt.savefig('C_Accuracy_model_1.png')

plt.subplot(1,2,2)
plt.plot(epochs,hist.history['loss'],label='Train Losss')
plt.plot(epochs,hist.history['val_loss'],label='Val Loss')
plt.title('CIFAR-Model-1')


plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('C_Loss_model_1.png')
plt.show()

"""## Model 2"""

# Adding more conv layers
model1 = Sequential()
model1.add(Conv2D(filters = 8, kernel_size = (3,3), activation='relu',padding="same",input_shape=Input_shape))
model1.add(Conv2D(filters = 16, kernel_size = (3,3), activation='relu',padding="same"))
model1.add(MaxPooling2D(2,2))

model1.add(Conv2D(filters = 32, kernel_size = (3,3), activation='relu',padding="same"))
model1.add(Conv2D(filters = 64, kernel_size = (3,3), activation='relu',padding="same"))
model1.add(MaxPooling2D(2,2))

model1.add(Conv2D(filters = 128, kernel_size = (3,3), activation='relu',padding="same"))
model1.add(Conv2D(filters = 256, kernel_size = (3,3), activation='relu',padding="same"))
model1.add(MaxPooling2D(2,2))



model1.add(Flatten())

model1.add(Dense(units =400,activation='relu'))
model1.add(Dense(units = 200,activation='relu'))

model1.add(Dense(units = 10,activation='softmax'))

model1.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc'])
model1.summary()

hist1 = model1.fit(xtrain,ytrain,batch_size= 32 ,epochs = 50,verbose=1,shuffle = True,validation_data=(xtest,ytest))

best=max(hist1.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score = model1.evaluate(xtest,ytest)
model1.save('model1.h5')

#Plotting data 
epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,hist1.history['acc'],label='Train Accuracy')
plt.plot(epochs,hist1.history['val_acc'],label='Test Accuracy')
plt.title('CIFAR-Model-2')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('C_Accuracy_model_2.png')

plt.subplot(1,2,2)
plt.plot(epochs,hist1.history['loss'],label='Train Losss')
plt.plot(epochs,hist1.history['val_loss'],label='Val Loss')
plt.title('CIFAR-Model-2')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('C_Loss_model_2.png')
plt.show()





"""## Model 3 using Dropout

"""

# Adding Dropout 
model2 = Sequential()
model2.add(Conv2D(filters = 8, kernel_size = (3,3), activation='relu',padding="same",input_shape=Input_shape))
model2.add(Conv2D(filters = 16, kernel_size = (3,3), activation='relu',padding="same"))
model2.add(MaxPooling2D(2,2))
model2.add(Dropout(0.3))

model2.add(Conv2D(filters = 32, kernel_size = (3,3), activation='relu',padding="same"))
model2.add(Conv2D(filters = 64, kernel_size = (3,3), activation='relu',padding="same"))
model2.add(MaxPooling2D(2,2))
model2.add(Dropout(0.3))

model2.add(Conv2D(filters = 128, kernel_size = (3,3), activation='relu',padding="same"))
model2.add(Conv2D(filters = 256, kernel_size = (3,3), activation='relu',padding="same"))
model2.add(MaxPooling2D(2,2))
model2.add(Dropout(0.3))


model2.add(Flatten())
model2.add(Dense(units =400,activation='relu'))
model2.add(Dense(units = 200,activation='relu'))
model2.add(Dense(units = 10,activation='softmax'))

model2.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc'])
model2.summary()

hist2 = model2.fit(xtrain,ytrain,batch_size= 32 ,epochs = 50,verbose=1,shuffle = True,validation_data=(xtest,ytest))

best=max(hist2.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score = model2.evaluate(xtest,ytest)
model2.save('model2.h5')

#Plotting Data
epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)

plt.plot(epochs,hist2.history['acc'],label='Train Accuracy')
plt.plot(epochs,hist2.history['val_acc'],label='Val Accuracy')
plt.title('CIFAR-Model-3')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('C_Accuracy_model_3.png')

plt.subplot(1,2,2)
plt.plot(epochs,hist2.history['loss'],label='Train Losss')
plt.plot(epochs,hist2.history['val_loss'],label='Val Loss')
plt.title('CIFAR-Model-3')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('C_Loss_model_3.png')
plt.show()

"""## Model 4 using Batch Normalization"""

# Adding Batch Normalization
model3 = Sequential()
model3.add(Conv2D(filters = 8, kernel_size = (3,3), activation='relu',padding="same",input_shape=Input_shape))
model3.add(BatchNormalization())
model3.add(Conv2D(filters = 16, kernel_size = (3,3), activation='relu',padding="same"))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(2,2))
model3.add(Dropout(0.3))

model3.add(Conv2D(filters = 32, kernel_size = (3,3), activation='relu',padding="same"))
model3.add(BatchNormalization())
model3.add(Conv2D(filters = 64, kernel_size = (3,3), activation='relu',padding="same"))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(2,2))
model3.add(Dropout(0.3))

model3.add(Conv2D(filters = 128, kernel_size = (3,3), activation='relu',padding="same"))
model3.add(BatchNormalization())
model3.add(Conv2D(filters = 256, kernel_size = (3,3), activation='relu',padding="same"))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(2,2))
model3.add(Dropout(0.3))

model3.add(Conv2D(filters = 512, kernel_size = (3,3), activation='relu',padding="same"))
model3.add(BatchNormalization())
model3.add(Conv2D(filters = 1024, kernel_size = (3,3), activation='relu',padding="same"))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(2,2))
model3.add(Dropout(0.3))

model3.add(Flatten())

model3.add(Dense(units =400,activation='relu'))
model3.add(Dense(units = 200,activation='relu'))

model3.add(Dense(units = 10,activation='softmax'))

model3.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model3.summary()

hist3 = model3.fit(xtrain,ytrain,batch_size= 32 ,epochs = 50,verbose=1,shuffle = True,validation_data=(xtest,ytest))

max(hist3.history['val_accuracy'])

best=max(hist3.history['val_accuracy']) *100
print('Best Accuracy = '+str(round(best,2)))

score = model3.evaluate(xtest,ytest)
model3.save('model3.h5')

# Plotting Data
epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,hist3.history['accuracy'],label='Train Accuracy')
plt.plot(epochs,hist3.history['val_accuracy'],label='Val Accuracy')
plt.title('CIFAR-Model-4')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('C_Accuracy_model_4.png')

plt.subplot(1,2,2)
plt.plot(epochs,hist3.history['loss'],label='Train Losss')
plt.plot(epochs,hist3.history['val_loss'],label='Val Loss')
plt.title('CIFAR-Model-4')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('C_Loss_model_4.png')
plt.show()

# Comparing 4 models
epochs = range(1,51)
plt.figure(figsize=(10,6))


plt.plot(epochs,hist.history['val_accuracy'],label='Model-1')
plt.plot(epochs,hist1.history['val_acc'],label='Model-2')
plt.plot(epochs,hist2.history['val_acc'],label='Model-3')
plt.plot(epochs,hist3.history['val_accuracy'],label='Model-4')
plt.title('CIFAR')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('Comp.png')

"""## Prediction

"""

# Using model 3 for predictions
predicted_class =  np.argmax(model3.predict(xtest),axis=1)
predicted_class

ytest = ytest.argmax(1)
ytest

w = 7
l = 7
fig,axes = plt.subplots(l,w,figsize = (12,12))
axes = axes.ravel();

for i in np.arange(0,l*w):
    axes[i].imshow(xtest[i])
    axes[i].set_title("Prediction={}\nTrue={}".format(predicted_class[i],ytest[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace = 1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(ytest,predicted_class)
cm

plt.figure(figsize=(10,10))
sns.heatmap(cm,annot = True)





"""# Image Classification on Fashion MNIST

###Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. 

###Each example is a 28x28 grayscale image, associated with a label from 10 classes.

## Data Loading
"""

from keras.datasets import fashion_mnist

(xtrain_fm,ytrain_fm),(xtest_fm,ytest_fm)=fashion_mnist.load_data()

"""## Data Preprocessing"""

input_shape=(28,28,1)

xtrain_fm=xtrain_fm/255
xtrain_fm=xtrain_fm.astype(np.float)

xtest_fm=xtest_fm/255
xtest_fm=xtest_fm.astype(np.float)

xtrain_fm=xtrain_fm.reshape(60000,28,28,1)
xtest_fm=xtest_fm.reshape(10000,28,28,1)

ytrain_fm = keras.utils.to_categorical(ytrain_fm,10)
ytest_fm = keras.utils.to_categorical(ytest_fm,10)

"""## Model Training

"""

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

"""## Model 1"""

model_FM = Sequential()

model_FM.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model_FM.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model_FM.add(MaxPooling2D(pool_size=(2,2)))

model_FM.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model_FM.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model_FM.add(MaxPooling2D(pool_size=(2,2)))



model_FM.add(Flatten())
model_FM.add(Dense(units=1024,activation='relu'))
model_FM.add(Dense(units=128,activation='relu'))

model_FM.add(Dense(units=10,activation='softmax'))

model_FM.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc']) 
model_FM.summary()

h=model_FM.fit(xtrain_fm,ytrain_fm,validation_data=(xtest_fm,ytest_fm),epochs=50,verbose=1,batch_size=batchsize,shuffle=True)

best=max(h.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score_FM = model_FM.evaluate(xtest_fm,ytest_fm)
model_FM.save('FM_model1.h5')

epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,h.history['acc'],label='Train Accuracy')
plt.plot(epochs,h.history['val_acc'],label='Test Accuracy')
plt.title('FM-Model-1')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('FM_Accuracy_model_1.png')


plt.subplot(1,2,2)
plt.plot(epochs,h.history['loss'],label='Train Losss')
plt.plot(epochs,h.history['val_loss'],label='Test Loss')
plt.title('FM-Model-1')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('FM_Loss_model_1.png')
plt.show()

"""## Model 2 """

model_FM1 = Sequential()

model_FM1.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model_FM1.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model_FM1.add(MaxPooling2D(pool_size=(2,2)))


model_FM1.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model_FM1.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model_FM1.add(MaxPooling2D(pool_size=(2,2)))


model_FM1.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model_FM1.add(MaxPooling2D(pool_size=(2,2)))



model_FM1.add(Flatten())
model_FM1.add(Dense(units=1024,activation='relu'))
model_FM1.add(Dense(units=128,activation='relu'))

model_FM1.add(Dense(units=10,activation='softmax'))

model_FM1.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc']) 
model_FM1.summary()

h1=model_FM1.fit(xtrain_fm,ytrain_fm,validation_data=(xtest_fm,ytest_fm),epochs=50,verbose=1,batch_size=batchsize,shuffle=True)

best=max(h1.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score_FM1 = model_FM1.evaluate(xtest_fm,ytest_fm)
model_FM1.save('FM_model2.h5')

epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,h1.history['acc'],label='Train Accuracy')
plt.plot(epochs,h1.history['val_acc'],label='Test Accuracy')
plt.title('FM-Model-2')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('FM_Accuracy_model_2.png')


plt.subplot(1,2,2)
plt.plot(epochs,h1.history['loss'],label='Train Losss')
plt.plot(epochs,h1.history['val_loss'],label='Test Loss')
plt.title('FM-Model-2')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('FM_Loss_model_2.png')
plt.show()

"""## Model 3 using Dropout"""

model_FM2 = Sequential()

model_FM2.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model_FM2.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model_FM2.add(MaxPooling2D(pool_size=(2,2)))
model_FM2.add(Dropout(0.25))

model_FM2.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model_FM2.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model_FM2.add(MaxPooling2D(pool_size=(2,2)))
model_FM2.add(Dropout(0.25))

model_FM2.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model_FM2.add(MaxPooling2D(pool_size=(2,2)))
model_FM2.add(Dropout(0.25))


model_FM2.add(Flatten())
model_FM2.add(Dense(units=1024,activation='relu'))
model_FM2.add(Dense(units=128,activation='relu'))

model_FM2.add(Dense(units=10,activation='softmax'))

model_FM2.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc']) 
model_FM2.summary()

h2=model_FM2.fit(xtrain_fm,ytrain_fm,validation_data=(xtest_fm,ytest_fm),epochs=50,verbose=1,batch_size=batchsize,shuffle=True)

best=max(h2.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score_FM2 = model_FM2.evaluate(xtest_fm,ytest_fm)
model_FM2.save('FM_model3.h5')

epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,h2.history['acc'],label='Train Accuracy')
plt.plot(epochs,h2.history['val_acc'],label='Test Accuracy')
plt.title('FM-Model-3')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('FM_Accuracy_model_3.png')


plt.subplot(1,2,2)
plt.plot(epochs,h2.history['loss'],label='Train Losss')
plt.plot(epochs,h2.history['val_loss'],label='Test Loss')
plt.title('FM-Model-3')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('FM_Loss_model_3.png')
plt.show()

"""## Model 4 using Batch Normalization

"""

model_FM3 = Sequential()

model_FM3.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model_FM3.add(BatchNormalization())
model_FM3.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model_FM3.add(BatchNormalization())
model_FM3.add(MaxPooling2D(pool_size=(2,2)))
model_FM3.add(Dropout(0.25))

model_FM3.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model_FM3.add(BatchNormalization())
model_FM3.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model_FM3.add(BatchNormalization())
model_FM3.add(MaxPooling2D(pool_size=(2,2)))
model_FM3.add(Dropout(0.25))

model_FM3.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model_FM3.add(BatchNormalization())
model_FM3.add(MaxPooling2D(pool_size=(2,2)))
model_FM3.add(Dropout(0.25))


model_FM3.add(Flatten())
model_FM3.add(Dense(units=1024,activation='relu'))
model_FM3.add(Dense(units=128,activation='relu'))

model_FM3.add(Dense(units=10,activation='softmax'))

model_FM3.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc']) 
model_FM3.summary()

h3=model_FM3.fit(xtrain_fm,ytrain_fm,validation_data=(xtest_fm,ytest_fm),epochs=50,verbose=1,batch_size=batchsize,shuffle=True)

best=max(h3.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score_FM3 = model_FM3.evaluate(xtest_fm,ytest_fm)
model_FM3.save('FM_model3.h5')

epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,h3.history['acc'],label='Train Accuracy')
plt.plot(epochs,h3.history['val_acc'],label='Test Accuracy')
plt.title('FM-Model-4')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('FM_Accuracy_model_4.png')


plt.subplot(1,2,2)
plt.plot(epochs,h3.history['loss'],label='Train Losss')
plt.plot(epochs,h3.history['val_loss'],label='Test Loss')
plt.title('FM-Model-4')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('FM_Loss_model_4.png')
plt.show()

epochs = range(1,51)
plt.figure(figsize=(10,6))


plt.plot(epochs,h.history['val_acc'],label='Model-1')
plt.plot(epochs,h1.history['val_acc'],label='Model-2')
plt.plot(epochs,h2.history['val_acc'],label='Model-3')
plt.plot(epochs,h3.history['val_acc'],label='Model-4')
plt.title('Fashion_MNIST')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim([0.8,1.0])
plt.grid()
plt.legend()
plt.savefig('Comp1.png')

"""## Prediction"""

predicted_class =  np.argmax(model_FM3.predict(xtest_fm),axis=1)
predicted_class

ytest_fm = ytest_fm.argmax(1)
ytest_fm

w = 7
l = 7
fig,axes = plt.subplots(l,w,figsize = (12,12))
axes = axes.ravel();

for i in np.arange(0,l*w):
    axes[i].imshow(xtest_fm[i][:,:,0], cmap='gray')
    axes[i].set_title("Prediction={}\nTrue={}".format(predicted_class[i],ytest_fm[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace = 1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(ytest_fm,predicted_class)
cm

plt.figure(figsize=(10,10))
sns.heatmap(cm,annot = True)



"""#Image Classification with MNIST Dataset

###The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.

###It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

###There 50000 Training samples and 10000 Testing Samples
"""

# Importing Libaries
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten, BatchNormalization
from keras.layers import Conv2D , MaxPooling2D
from keras import backend as k
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

"""## Data Loading"""

from keras.datasets import mnist

# Load dataset
(xtrain_mn , ytrain_mn) , (xtest_mn , ytest_mn)=mnist.load_data()

"""## Data Preprocessing"""

xtrain_mn=xtrain_mn.reshape(xtrain_mn.shape[0],28,28,1)
xtest_mn=xtest_mn.reshape(xtest_mn.shape[0],28,28,1)
input_shape=(28,28,1)

ytrain_mn=keras.utils.to_categorical(ytrain_mn, 10)
ytest_mn=keras.utils.to_categorical(ytest_mn,10)

xtrain_mn = xtrain_mn.astype('float32')
xtest_mn = xtest_mn.astype('float32')

xtrain_mn=xtrain_mn/255
xtest_mn=xtest_mn/255

"""## Model Training"""

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

"""## Model 1"""

model_MN=Sequential()
model_MN.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model_MN.add(Conv2D(16,(3,3),activation='relu'))
model_MN.add(MaxPooling2D(pool_size=(2,2)))
model_MN.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model_MN.add(Conv2D(64,(3,3),activation='relu'))
model_MN.add(MaxPooling2D(pool_size=(2,2)))

model_MN.add(Flatten())
model_MN.add(Dense(256, activation='relu'))
model_MN.add(Dense(128, activation='relu'))
model_MN.add(Dense(10, activation='softmax'))

model_MN.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc'])
model_MN.summary()

# Model training
history=model_MN.fit(xtrain_mn,ytrain_mn , batch_size=32,epochs=50,verbose=1,validation_data=(xtest_mn,ytest_mn))

best=max(history.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score_MN=model_MN.evaluate(xtest_mn,ytest_mn)
model_MN.save('MN_model1.h5')

epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,history.history['acc'],label='Train Accuracy')
plt.plot(epochs,history.history['val_acc'],label='Test Accuracy')
plt.title('MNIST-Model-1')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('MN_Accuracy_model_1.png')


plt.subplot(1,2,2)
plt.plot(epochs,history.history['loss'],label='Train Losss')
plt.plot(epochs,history.history['val_loss'],label='Test Loss')
plt.title('MNIST-Model-1')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('MN_Loss_model_1.png')
plt.show()

"""## Model 2"""

# Prediction

model_MN1=Sequential()
model_MN1.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model_MN1.add(Conv2D(16,(3,3),activation='relu'))
model_MN1.add(MaxPooling2D(pool_size=(2,2)))
model_MN1.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model_MN1.add(Conv2D(64,(3,3),activation='relu'))
model_MN1.add(MaxPooling2D(pool_size=(2,2)))
model_MN1.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model_MN1.add(MaxPooling2D(pool_size=(2,2)))

model_MN1.add(Flatten())
model_MN1.add(Dense(256, activation='relu'))
model_MN1.add(Dense(128, activation='relu'))

model_MN1.add(Dense(10, activation='softmax'))

model_MN1.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc']) 
model_MN1.summary()

# Model training
history1=model_MN1.fit(xtrain_mn,ytrain_mn , batch_size=32,epochs=50,verbose=1,validation_data=(xtest_mn,ytest_mn))

best=max(history1.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score_MN1=model_MN1.evaluate(xtest_mn,ytest_mn)
model_MN1.save('MN_model2.h5')

epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,history1.history['acc'],label='Train Accuracy')
plt.plot(epochs,history1.history['val_acc'],label='Test Accuracy')
plt.title('MNIST-Model-2')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('MN_Accuracy_model_2.png')


plt.subplot(1,2,2)
plt.plot(epochs,history1.history['loss'],label='Train Losss')
plt.plot(epochs,history1.history['val_loss'],label='Test Loss')
plt.title('MNIST-Model-2')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('MN_Loss_model_2.png')
plt.show()



"""## Model 3 using Dropout"""

model_MN2=Sequential()
model_MN2.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model_MN2.add(Conv2D(16,(3,3),activation='relu'))
model_MN2.add(MaxPooling2D(pool_size=(2,2)))
model_MN2.add(Dropout(0.25))
model_MN2.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model_MN2.add(Conv2D(64,(3,3),activation='relu'))
model_MN2.add(MaxPooling2D(pool_size=(2,2)))
model_MN2.add(Dropout(0.25))
model_MN2.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model_MN2.add(MaxPooling2D(pool_size=(2,2)))
model_MN2.add(Dropout(0.25))

model_MN2.add(Flatten())
model_MN2.add(Dense(256, activation='relu'))
model_MN2.add(Dense(128, activation='relu'))

model_MN2.add(Dense(10, activation='softmax'))

model_MN2.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc']) 
model_MN2.summary();

# Model training
history2=model_MN2.fit(xtrain_mn,ytrain_mn , batch_size=32,epochs=50,verbose=1,validation_data=(xtest_mn,ytest_mn))



best=max(history2.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score_MN2=model_MN2.evaluate(xtest_mn,ytest_mn)
model_MN2.save('MN_model3.h5')

epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,history2.history['acc'],label='Train Accuracy')
plt.plot(epochs,history2.history['val_acc'],label='Test Accuracy')
plt.title('MNIST-Model-3')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('MN_Accuracy_model_3.png')


plt.subplot(1,2,2)
plt.plot(epochs,history2.history['loss'],label='Train Losss')
plt.plot(epochs,history2.history['val_loss'],label='Test Loss')
plt.title('MNIST-Model-3')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('MN_Loss_model_3.png')
plt.show()

"""## Model 4 using Batch Normalization"""

model_MN3=Sequential()
model_MN3.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model_MN3.add(BatchNormalization())
model_MN3.add(Conv2D(16,(3,3),activation='relu'))
model_MN3.add(BatchNormalization())
model_MN3.add(MaxPooling2D(pool_size=(2,2)))
model_MN3.add(Dropout(0.25))
model_MN3.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model_MN3.add(BatchNormalization())
model_MN3.add(Conv2D(64,(3,3),activation='relu'))
model_MN3.add(BatchNormalization())
model_MN3.add(MaxPooling2D(pool_size=(2,2)))
model_MN3.add(Dropout(0.25))
model_MN3.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model_MN3.add(BatchNormalization())
model_MN3.add(MaxPooling2D(pool_size=(2,2)))
model_MN3.add(Dropout(0.25))

model_MN3.add(Flatten())
model_MN3.add(Dense(256, activation='relu'))
model_MN3.add(Dense(128, activation='relu'))

model_MN3.add(Dense(10, activation='softmax'))

model_MN3.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc']) 
model_MN3.summary()

# Model training
history3=model_MN3.fit(xtrain_mn,ytrain_mn , batch_size=32,epochs=50,verbose=1,validation_data=(xtest_mn,ytest_mn))



best=max(history3.history['val_acc']) *100
print('Best Accuracy = '+str(round(best,2)))

score_MN3=model_MN3.evaluate(xtest_mn,ytest_mn)
model_MN3.save('MN_model4.h5')

epochs = range(1,51)
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
plt.plot(epochs,history3.history['acc'],label='Train Accuracy')
plt.plot(epochs,history3.history['val_acc'],label='Test Accuracy')
plt.title('MNIST-Model-4')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('MN_Accuracy_model_4.png')


plt.subplot(1,2,2)
plt.plot(epochs,history3.history['loss'],label='Train Losss')
plt.plot(epochs,history3.history['val_loss'],label='Test Loss')
plt.title('MNIST-Model-4')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend()
plt.savefig('MN_Loss_model_4.png')
plt.show()

epochs = range(1,51)
plt.figure(figsize=(10,6))


plt.plot(epochs,history.history['val_acc'],label='Model-1')
plt.plot(epochs,history1.history['val_acc'],label='Model-2')
plt.plot(epochs,history2.history['val_acc'],label='Model-3')
plt.plot(epochs,history3.history['val_acc'],label='Model-4')
plt.title('MNIST')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim([0.9,1.0])
plt.grid()
plt.legend()
plt.savefig('Comp2.png')

"""##Prediction"""

predicted_class =  np.argmax(model_MN.predict(xtest_mn),axis=1)
predicted_class

ytest_mn = ytest_mn.argmax(1)
ytest_mn

w = 7
l = 7
fig,axes = plt.subplots(l,w,figsize = (12,12))
axes = axes.ravel();

for i in np.arange(0,l*w):
    axes[i].imshow(xtest_mn[i][:,:,0], cmap='gray')
    axes[i].set_title("Prediction={}\nTrue={}".format(predicted_class[i],ytest_mn[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace = 1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(ytest_mn,predicted_class)
cm

plt.figure(figsize=(10,10))
sns.heatmap(cm,annot = True)



# Comparision between models performance on datasets
epochs = range(1,51)
plt.figure(figsize=(10,6))


plt.plot(epochs,hist3.history['val_accuracy'],label='CIFAR-Model')
plt.plot(epochs,h3.history['val_acc'],label='F-MNIST-Model')
plt.plot(epochs,history3.history['val_acc'],label='MNIST-Model')
plt.title('Comparision')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.grid()
plt.legend()
plt.savefig('Comp3.png')

