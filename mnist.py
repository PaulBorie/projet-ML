#!/usr/bin/env python3

import naivebayes
import knn
import freeman
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


##Binary representations

x_train_binary = freeman.convert_binary(x_train)
x_test_binary = freeman.convert_binary(x_test)

## Flatten binary pixel matrix 

x_train_binary = x_train_binary.reshape(60000, 784)
x_test_binary = x_test_binary.reshape(10000, 784)
x_test_binary_small = x_test_binary[0:20, :]
y_test_binary_small = y_test[0:20]

## Flatten non binary images
x_train=x_train.reshape(60000, 784)
x_test=x_test.reshape(10000,784)
x_test_small = x_test[0:200, :]
y_test_small = y_test[0:200]


## Freeman representation

def naive_bayes():
    gaussian_bayes = naivebayes.Bayes()
    gaussian_bayes.fit(x_train, y_train)
    gaussian_bayes.predict(x_test_small, y_test_small)


def naive_bayes_binary():
    gaussian_bayes = naivebayes.Bayes()
    gaussian_bayes.fit(x_train_binary, y_train)
    gaussian_bayes.predict2(x_test_small, y_test_small)


def k_nearest_neighbor():
    k_nn = knn.Knn(3)
    k_nn.predict_euclidian_dist(x_train, y_train, x_test_small, y_test_small)

def k_nearest_neighbor_binary():
    k_nn = knn.Knn(3)
    k_nn.predict_euclidian_dist(x_train_binary, y_train, x_test_binary_small, y_test_binary_small)

def k_nearest_neighbor_levenstein():
    k_nn = knn.Knn(3)
    #k_nn.predict_levenshtein_dist(x_train_freeman, y_train, x_test_freeman, y_test_binary_small )
        

if __name__ == "__main__":  
    k_nearest_neighbor_binary()
    #naive_bayes()
    #k_nearest_neighbor_levenstein()
    #naive_bayes_binary()