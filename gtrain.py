import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255
X_train_flatten = X_train.reshape(len(X_train), 28*28)
X_test_flatten = X_test.reshape(len(X_test), 28*28)

model = keras.Sequential([keras.layers.Dense(100, input_shape =(784,),activation = 'relu'), keras.layers.Dense(10, activation = 'sigmoid')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train_flatten, y_train, epochs = 7)
model.evaluate(X_test_flatten, y_test)
y_predicted = model.predict(X_test_flatten)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
# cm = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)
# plt.figure(figsize = (10,7))
# sn.heatmap(cm, annot = True, fmt = 'd')
# plt.xlabel('predicted')
# plt.ylabel('truth')

model.save('./mnist.h5')
print("Saving the model as mnist.h5")
