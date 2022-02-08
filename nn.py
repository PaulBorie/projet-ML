#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D,Dropout
import freeman
from tensorflow.keras.optimizers import SGD


class Neuralnetwork:

	def __init__(self):

		(self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

		#Représentation bianire
		self.X_train = freeman.convert_binary(self.X_train)
		self.X_test = freeman.convert_binary(self.X_test)
				
	def fit(self, nb_epoch):

		
		# Convert y_train into one-hot format 
		temp = []
		for i in range(len(self.y_train)):
			temp.append(to_categorical(self.y_train[i], num_classes=10))
			
		y_train = np.array(temp)

		# Convert y_test into one-hot format 
		temp = []
		for i in range(len(self.y_test)):
			temp.append(to_categorical(self.y_test[i], num_classes=10))

		y_test = np.array(temp)

		self.model = Sequential()
		self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
		self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Flatten())
		self.model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
		self.model.add(Dense(10, activation='softmax'))
		# compilation du modèle
		opt = SGD(learning_rate=0.01, momentum=0.9)
		self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

		# Entrainement du modèle
		self.model.fit(self.X_train, y_train, epochs=10,batch_size=32, validation_data=(self.X_test,y_test))
		self.model.save("digit_recognition_optimized.h5")


	def predict(self, X_test):
		predictions = self.model.predict(X_test)
		predictions = np.argmax(predictions, axis=1)
		return predictions

def main():
	nn = Neuralnetwork()
	nn.fit(5)
	nn.predict(nn.X_test)

if __name__ == "__main__":  
    main()