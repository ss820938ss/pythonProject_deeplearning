from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

Data_set = np.loadtxt("./data/ThoraricSurgery.csv", delimiter=",")

x = Data_set[:, 0:17]
y = Data_set[:, 17]

model = Sequential()

model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, epochs=100, batch_size=10)

