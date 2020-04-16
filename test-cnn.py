import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
test = pd.read_csv("./test.txt")
test_X = np.array(test.iloc[:, 1:])
test_Y = np.array(test.iloc[:, 0])
scaler = MinMaxScaler()
scaler.fit(test_X)
model = tf.keras.models.load_model('saved_model/bp-model')
test_X_normalized = scaler.transform(test_X)
test_X_normalized = np.nan_to_num(test_X_normalized).reshape(len(test_X_normalized), 72)
test_Y = test_Y//3
test_loss, test_acc = model.evaluate(test_X_normalized,  test_Y, verbose=2)

print('\nTest accuracy:', test_acc)
