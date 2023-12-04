### Developer Umut Kaan Eren September 2023.
### Resources used to write these codes:
### Coursera Deep Learning Specialization Course Trigger Word Detection Assignment -Andrew Ng
### Youtube Channel for general idea of the model : https://www.youtube.com/watch?v=yv_WVwr6OkI
###                                                 https://www.youtube.com/watch?v=0fn7pj7Dutc
###                                                 https://www.youtube.com/watch?v=NITIefkRae0
### Github Open Sources
#IMPORTS
from os import listdir
from os.path import isdir, join
from tensorflow.keras import layers, models
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


## calling training, validation and test data
feature_sets = np.load('all_labels_mfcc_sets.npz')
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']

# CNN for TF expects (batch, height, width, channels) but our data set format is (batch, height, width)
# So we reshape the input tensors with a "color" channel of 1
x_train = x_train.reshape(x_train.shape[0],
                          x_train.shape[1],
                          x_train.shape[2],
                          1)
x_val = x_val.reshape(x_val.shape[0],
                      x_val.shape[1],
                      x_val.shape[2],
                      1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)

#Model
model = models.Sequential()
model.add(layers.Conv2D(32,
                        (2, 2),
                        activation='relu',
                        input_shape=sample_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


# Classifier
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.7))
model.add(layers.Dense(1, activation='sigmoid'))

#Compiling the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

#Training
history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=100,
                    validation_data=(x_val, y_val))

# Plot results for training-validation accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Saving the model for future use
model.save('cnn_model_10.keras')