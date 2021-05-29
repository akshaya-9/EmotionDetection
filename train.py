
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from keras.utils import np_utils  
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D 
from keras.layers import BatchNormalization
from keras.optimizers import Adam 
from keras.utils import np_utils  
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import model_from_json
import cv2,time
import warnings
warnings.filterwarnings('ignore')


data  = pd.read_csv('fer2013.csv')

x_train =[]
y_train =[]
x_test = []
y_test =[]

for index,row in data.iterrows():
  k = row['pixels'].split(" ")
  if row['Usage'] == 'Training':
    x_train.append(np.array(k))
    y_train.append(row['emotion'])
  elif row['Usage'] == 'PrivateTest' or row['Usage'] == 'PublicTest' :
    x_test.append(np.array(k))
    y_test.append(row['emotion'])

x_train = np.array(x_train,'float32')
y_train = np.array(y_train,'float32')
x_test = np.array(x_test,'float32')
y_test =np.array(y_test,'float32')


x_train -= np.mean(x_train, axis=0)  
x_train /= np.std(x_train, axis=0)  
  
x_test -= np.mean(x_test, axis=0)  
x_test /= np.std(x_test, axis=0)  

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)  
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)  

y_train =np_utils.to_categorical(y_train,num_classes=7)
y_test = np_utils.to_categorical(y_test,num_classes=7)


model = Sequential()
#layer 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1),padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# layer 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#layer 3
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#layer4 
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

#fully connected layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
import tensorflow as tf
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=8, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)


history = model.fit(x_train,y_train,batch_size=120,epochs=100,verbose=1,validation_data=(x_test,y_test), callbacks=[early_stopping], shuffle=True)


scores = model.evaluate(x_test, y_test, verbose=0)
print(scores)
print("Testing Accuracy: %.2f%%" % (scores[1]*100))


def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history, 'loss') 
plot_metric(history, 'accuracy')

model_json = model.to_json()
with open("model.json", "w") as json_file:
 json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
