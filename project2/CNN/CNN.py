import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy.io as sio
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load data
training_set = sio.loadmat('trainingset.mat')
testing_set = sio.loadmat('testingset.mat')
x_train = training_set['trainingset']
x_test = testing_set['testingset']
train_num = len(x_train)
test_num = len(x_test)

# build convolution neural network model
CNN_model = keras.Sequential(
    [layers.Conv2D(20, 5, strides=(1, 1), padding='same', activation='relu',
                   kernel_regularizer=keras.regularizers.l2(0.001),
                   kernel_initializer=keras.initializers.GlorotNormal(),
                   input_shape=(32, 32, 1)),
     layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     layers.Conv2D(50, 5, strides=(1, 1), padding='same', activation='relu',
                   kernel_regularizer=keras.regularizers.l2(0.001),
                   kernel_initializer=keras.initializers.GlorotNormal(),
                   ),
     layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     layers.Flatten(),
     layers.Dense(500, activation='relu'),
     layers.Dense(26, activation='softmax')]
)

CNN_model.compile(optimizer=keras.optimizers.RMSprop(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

# adjust data
y_train = [int(i / 119) for i in range(train_num)]
y_test = [int(i / 51) for i in range(test_num)]
x_train = x_train.reshape(train_num, 32, 32, 1)
x_test = x_test.reshape(test_num, 32, 32, 1)
y_train = np.array(y_train)
y_test = np.array(y_test)

# train and fit the model and collect data
epoch_times = 31
train_history = CNN_model.fit(x_train, y_train, batch_size=64, epochs=epoch_times, validation_data=(x_test, y_test))
train_accuracy = train_history.history['accuracy']
train_loss = train_history.history['loss']
test_accuracy = train_history.history['val_accuracy']
test_loss = train_history.history['val_loss']

train_accuracy_sample = [train_accuracy[i * 5] for i in range(6)]
train_loss_sample = [train_loss[i * 5] for i in range(6)]
test_accuracy_sample = [test_accuracy[i * 5] for i in range(6)]
test_loss_sample = [test_loss[i * 5] for i in range(6)]
print('train accuracy', train_accuracy_sample)
print('train loss', train_loss_sample)
print('test accuracy', test_accuracy_sample)
print('test loss', test_loss_sample)
test_scores = CNN_model.evaluate(x_test, y_test, verbose=2)
train_scores = CNN_model.evaluate(x_train, y_train, verbose=2)
print("Final testing loss:", test_scores[0], " Final testing accuracy:", test_scores[1])
print("Fina training loss:", train_scores[0], " Final training accuracy:", train_scores[1])

# draw loss figure
plt.figure(1)
x = [i for i in range(epoch_times)]
plt.plot(x, train_loss, label='Training loss')
plt.plot(x, test_loss, label='Testing loss')
plt.xlabel('Epoch times')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title("Training and testing loss")
plt.grid()
plt.show()

# draw accuracy figure
plt.figure(2)
plt.plot(x, train_accuracy, label='Training accuracy')
plt.plot(x, test_accuracy, label='Testing accuracy')
plt.xlabel('Epoch times')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title("Training and testing accuracy")
plt.grid()
plt.show()
