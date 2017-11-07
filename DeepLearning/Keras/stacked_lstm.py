from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
xt = np.random.random((1000, timesteps, data_dim))
x_train = xt.astype(np.float32)
yt = np.random.random((1000, num_classes))
y_train = yt.astype(np.float32)

# Generate dummy validation data
xv = np.random.random((100, timesteps, data_dim))
x_val = xv.astype(np.float32)
yv = np.random.random((100, num_classes))
y_val = yv.astype(np.float32)

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
