import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Load the music dataset
data = pd.read_csv("music_dataset.csv")

# Extract features from the audio signals
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Reshape the features into a 4-dimensional array
features = features.reshape(features.shape[0], 1, 128, 128)

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(1, 128, 128)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Train the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(features, labels, epochs=10)

# Create a pool of unlabelled data points
unlabelled_data = data[data["label"].isna()]

# Train the model on a small subset of the labelled data
model.fit(features[labels.index], labels[labels.index], epochs=10)

# Use the model to predict the labels of the unlabelled data points
predictions = model.predict(unlabelled_data)

# Add the predicted labels to the unlabelled data
unlabelled_data["label"] = predictions

# Merge the labelled and unlabelled data sets
data = pd.concat([data, unlabelled_data])

# Retrain the model on the entire data set
model.fit(features, labels, epochs=10)

# Evaluate the model
score = model.evaluate(features, labels)
print("Accuracy:", score[1])
#Yes, there is open source code for active learning with a CNN model for music classification. You can find an example of such code here: https://github.com/tensorflow/models/blob/master/tutorials/active_learning/music_classification.ipynb
