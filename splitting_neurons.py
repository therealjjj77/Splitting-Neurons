#Developed by Jeremiah Johnson - https://jjohnson-777.medium.com/ https://github.com/therealjjj77
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
num_neurons=2

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# define model architecture
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(num_neurons, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
# number of epochs doesn't matter because we'll be using the EarlyStopping callback
epochs = 1000

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
          callbacks=keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=.01, restore_best_weights=True))

# this gets a list of the weights (2 arrays) and biases (2 arrays)
weights=model.get_weights()

c=0
# loop through list of numpy arrays, when you are not sure on axis, use print(i.shape) to know what's under the hood
for i in weights:
    if c < 3:
        # repeat elements on layer 1 weights and biases
        if c < 2:
            layerpre = np.repeat( i, 2, axis=-1)
            #nudge biases
            if c==1:
                layerpre[::2] += 0.001
                layerpre[1::2] += -0.001
            weights[c] = layerpre

        # repeat elements on output layer weights and divide by two
        if c==2:
            layerpre = np.repeat(i, 2, axis=0)
            layerpre = layerpre/2
            weights[c] = layerpre
    #Note that we do not change output layer biases
    else:
        break
    c+=1

# create second model architecture, only difference is twice the neurons in the first layer
model2 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(num_neurons*2, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model2.summary()

model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Set the weights before training
model2.set_weights(weights)

model2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
           callbacks=keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=.005))