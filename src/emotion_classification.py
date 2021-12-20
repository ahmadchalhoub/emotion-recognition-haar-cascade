# Authors: Ahmad Chalhoub, Harjas Dadiyal, Geanna Perera

# Description: This is a CNN that takes in the output of a 
# face detection Haar Cascade and classifies the facial emotions 
# of the detected faces

# CLASSES:
######################
# 0 - Angry
# 1 - Disgust
# 2 - Fear
# 3 - Happy
# 4 - Sad
# 5 - Surprise
# 6 - Neutral
######################

# Import all required libraries
import pandas as pd
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers

# Read csv file of data
emotion_dataset = pd.read_csv(r"D:\School\Super_Senior\Deep_Learning\Project\emotion-recognition-haar-cascade\data\\fer2013.csv")
X_train = []
y_train = []
X_test = []
y_test = []

# Go through all elements of CSV files and store into arrays
for index, row in emotion_dataset.iterrows():
      pixel_values = row['pixels'].split(" ")
      if (row['Usage'] == 'Training'):
        X_train.append(np.array(pixel_values))
        y_train.append(row['emotion'])
      elif (row['Usage'] == 'PublicTest'):
        X_test.append(np.array(pixel_values))
        y_test.append(row['emotion'])

# Convert all arrays into numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Reshape training and testing data into required input dimensions, convert to float, and normalize
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_train = X_train.astype("float32") / 255

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
X_test = X_test.astype("float32") / 255

# Perform one-hot encoding on training and testing labels
y_train= to_categorical(y_train)
y_test = to_categorical(y_test)


# Declare CNN model architecture
emotion_model = models.Sequential()

# 2 Convolutional layers, each with 64 filters of size (3, 3), 'same' padding, activation function = relu
emotion_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))      
emotion_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))

# Max Pooling layer
emotion_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# 2 Convolutional layers, each with 128 filters of size (3, 3), 'same' padding, activation function = relu
emotion_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
emotion_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
emotion_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# 3 Convolutional layers, each with 256 filters of size (3, 3), 'same' padding, activation function = relu
emotion_model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
emotion_model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
emotion_model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
emotion_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten layer that flattens output of last Max Pooling layer into a 1D array
emotion_model.add(layers.Flatten())

# 2 Fully Connected layers, each with 4096 neurons and activation = relu
emotion_model.add(layers.Dense(4096, activation='relu'))
emotion_model.add(layers.Dense(4096, activation='relu'))

# Final layer with 7 neurons, each for a class, and activation = softmax
emotion_model.add(layers.Dense(7, activation='softmax'))

# Compile model with 'categorical_crossentropy' loss and Adam optimizer.
emotion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model using 17 epochs and batch size = 128
emotion_model = emotion_model.fit(X_train, y_train, epochs=17, batch_size=128,
          validation_data=(X_test, y_test), shuffle=True, verbose=1)


# Save trained model into disk so it can later be loaded and combined with Haar Cascade
model_json = emotion_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
emotion_model.save_weights("emotion_model_local_training.h5")
print("Saved model to disk")
