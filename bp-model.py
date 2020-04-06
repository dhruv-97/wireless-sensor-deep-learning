import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
li = []
for filename in glob.glob("Processed/*.txt"):
    df = pd.read_csv(filename, sep=',', header = None, index_col=None)
    li.append(df)
frame = pd.concat(li, axis = 0, ignore_index=True)
train, test = train_test_split(frame, test_size=0.2)
import numpy as np
train_X = np.array(train.iloc[:, 1:])
train_Y = np.array(train.iloc[:, 0])
test_X = np.array(test.iloc[:, 1:])
test_Y = np.array(test.iloc[:, 0])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_X)
train_X_normalized = scaler.transform(train_X)
train_X_normalized = np.nan_to_num(train_X_normalized)
scaler = MinMaxScaler()
scaler.fit(test_X)
test_X_normalized = scaler.transform(test_X)
test_X_normalized = np.nan_to_num(test_X_normalized)
# By diving by 3, we are classifying into 10 gases instead of gas along with fan speed
# Remove the 3 if you want to try 30 classes instead of 10
train_Y = train_Y//3
test_Y = test_Y//3
model = models.Sequential()
model.add(layers.Dense(36, activation='relu'))
model.add(layers.Dense(10)) #Replace 10 by 30 if doing for 30 classes
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_X_normalized, train_Y, validation_split = 0.3, epochs=2)
model.save('./saved_model/bp-model')
test_loss, test_acc = model.evaluate(test_X_normalized,  test_Y, verbose=2)

print('\nTest accuracy:', test_acc)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('bp-accuracy.png')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('bp-loss.png')
