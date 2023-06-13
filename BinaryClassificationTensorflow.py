import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from planar_utils import load_planar_dataset

# features are (x,y) coordinates and labels are 0 and 1
features, labels = load_planar_dataset()

# transposing the two ndarrays so their column headers are the features
labels = labels.transpose()
features = features.transpose()

# initializing tje neural network and its hidden layers
model = tf.keras.Sequential()
model.add(layers.Dense(4, input_shape=(2,), activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

# configures the model and starts the training
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.1), metrics=['accuracy'])
model.fit(features, labels, validation_split=0.2, epochs=200, batch_size=10)

# testing the model after learning is finished
predictions = model.predict(features).round()

# ploting the testing result
plt.scatter(features[:,0], features[:,1], c=predictions, s=40, cmap=plt.cm.Spectral)
plt.show()

# converting the model structure to human-readable text
config = model.to_json()

# saving qll the values from the model
weights = model.get_weights()
