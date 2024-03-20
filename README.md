# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:


### STEP 2:

### STEP 3:


## PROGRAM
```
### Name: ABISHAI K C
### Register Number: 212223240002

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(layers.Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(15,activation='relu'))
model.add(layers.Dense(10,activation = 'softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)


print(confusion_matrix(y_test,x_test_predictions))


print(classification_report(y_test,x_test_predictions))
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Abishai95141/mnist-classification/assets/139335314/6c0d0d57-a74d-4211-97d8-4eaf32064fcc)


### Classification Report

![image](https://github.com/Abishai95141/mnist-classification/assets/139335314/0d518de9-fc42-4299-9fb6-14c610d8cc1f)


### Confusion Matrix

![image](https://github.com/Abishai95141/mnist-classification/assets/139335314/0d40d9cd-807c-4a89-b339-a6a0f3217422)


### New Sample Data Prediction

#### Input
![th (2)](https://github.com/Abishai95141/mnist-classification/assets/139335314/83bf5666-8525-4c11-b124-a08c57720cec)

#### Output
![image](https://github.com/Abishai95141/mnist-classification/assets/139335314/ba1fa12e-264a-4f86-8509-7290a8693428)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
