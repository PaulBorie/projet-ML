#!/usr/bin/env python3

import numpy as np
from keras.datasets import mnist
import freeman

(x_train, y_train), (x_test, y_test) = mnist.load_data()

##Binary representation

x_train_binary = freeman.convert_binary(x_train)
x_test_binary = freeman.convert_binary(x_test)

## Flatten binary pixel matrix 

x_train_binary = x_train_binary.reshape(60000, 784)
x_test_binary = x_test_binary.reshape(10000, 784)
x_test_binary_small = x_test_binary[0:50, :]
y_test_binary_small = y_test[0:50]

class Knn:

    def __init__(self, k):
        self.k = k
    
    def mnist_euclidian_dist(self, x_test):
        preds = self.predict_euclidian_dist(x_train_binary, y_train, x_test, y_test_binary_small)
        print("preds: {}".format(preds[0][0]))
        return preds[0][0]
        
    
    def print_distances(self, distances, token1Length, token2Length):
        for t1 in range(token1Length + 1):
            for t2 in range(token2Length + 1):
                print(int(distances[t1][t2]), end=" ")
            print()
        print("\n")

    
    def levenshtein_distance_DP(self, token1, token2):
        distances = np.zeros((len(token1) + 1, len(token2) + 1))

        for t1 in range(len(token1) + 1):
            distances[t1][0] = t1

        for t2 in range(len(token2) + 1):
            distances[0][t2] = t2
        
   
        a = 0
        b = 0
        c = 0
        
        for t1 in range(1, len(token1) + 1):
            for t2 in range(1, len(token2) + 1):
                if (token1[t1-1] == token2[t2-1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]
                    
                    if (a <= b and a <= c):
                        distances[t1][t2] = a + 1
                    elif (b <= a and b <= c):
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1

        #self.print_distances(distances, len(token1), len(token2))
        return distances[len(token1)][len(token2)]

    def levenshtein_distance(self, row, test_data):
        return self.levenshtein_distance_DP(row, test_data )

    def predict_levenshtein_dist(self, x_train, y_train, x_test, y_test):
        preds = []
        nb_test_data = x_test.shape[0]
        for row in x_test:
            dists = np.apply_along_axis(self.levenshtein_distance, 1, x_train, row)
            dists = dists.reshape((dists.shape[0], 1))
            y_train = y_train.reshape((y_train.shape[0], 1))
            dists_with_classes = np.append(y_train, dists, axis=1)
            sorted_dists = dists_with_classes[dists_with_classes[:, 1].argsort()]
            k_nearest_neighboors = sorted_dists[:self.k, :]
            print(k_nearest_neighboors)
            knn_classes = k_nearest_neighboors[:,0].astype(int)
            pred = np.bincount(knn_classes).argmax()
            preds.append(pred)
        print(np.array(preds))
        good_pred_count = np.count_nonzero(preds == y_test)
        accuracy = good_pred_count / nb_test_data 
        print("accuracy {}".format(accuracy))
        return preds, accuracy 

    def predict_euclidian_dist(self, x_train, y_train, x_test, y_test):
        preds = []
        nb_test_data = x_test.shape[0]
        nb_train_data = x_train.shape[0]
        for row in x_test:
            row_tiled = np.tile(row, (nb_train_data, 1))
            dists = np.linalg.norm(x_train - row_tiled, axis=1)
            # On append 
            dists = dists.reshape((dists.shape[0], 1))
            y_train = y_train.reshape((y_train.shape[0], 1))
            dists_with_classes = np.append(y_train, dists, axis=1)
            sorted_dists = dists_with_classes[dists_with_classes[:, 1].argsort()]
            print(sorted_dists)
            print(sorted_dists)
            k_nearest_neighboors = sorted_dists[:self.k, :]
            print(k_nearest_neighboors)
            knn_classes = k_nearest_neighboors[:,0].astype(int)
            pred = np.bincount(knn_classes).argmax()
            preds.append(pred)
        print("preds : {}".format(np.array(preds)))
        # On compte le nombre de bonne prédictions
        good_pred_count = np.count_nonzero(preds == y_test)
        accuracy = good_pred_count / nb_test_data 
        print("accuracy {}".format(accuracy))
        return preds, accuracy   
        
    def predict_freeman_edit_distance(self, nb_imgs, nb_test_data):
        
        preds = []
        freemans = []

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train[0:nb_imgs, : , :]
        y_train = y_train[0:nb_imgs]

        x_test = x_test[0:nb_test_data, :, :]
        y_test = y_test[0:nb_test_data]

        freemans, to_delete_indexes = freeman.freeman_representation(x_train)
        for index in to_delete_indexes:
            y_train = np.delete(y_train, (index), axis=0 )

        x_test_freemans, to_delete_indexes_x_test = freeman.freeman_representation(x_test)
        for index in to_delete_indexes_x_test:
            y_test = np.delete(y_test, (index), axis=0 )
        
        print(len(x_test_freemans))
        print(y_test.shape)

        for row in x_test_freemans:
            dists = []
            for freeman_img in freemans:
                dist = self.levenshtein_distance(freeman_img, row)
                dists.append(dist)
            print(dists)

            dists = np.array(dists)
            dists = dists.reshape((dists.shape[0], 1))
            print("dists shape: {}".format(dists.shape))
            y_train = y_train.reshape((y_train.shape[0], 1))
            print("y_train.shape: {}".format(y_train.shape))
            dists_with_classes = np.append(y_train, dists, axis=1)
            print(dists_with_classes.shape)

            sorted_dists = dists_with_classes[dists_with_classes[:, 1].argsort()]
            print(sorted_dists)
            print(sorted_dists)
            k_nearest_neighboors = sorted_dists[:self.k, :]
            print(k_nearest_neighboors)
            knn_classes = k_nearest_neighboors[:,0].astype(int)
            pred = np.bincount(knn_classes).argmax()
            preds.append(pred)
        print("preds : {}".format(np.array(preds)))
        # On compte le nombre de bonne prédictions
        good_pred_count = np.count_nonzero(preds == y_test)
        accuracy = good_pred_count / nb_test_data 
        print("accuracy {}".format(accuracy))
        return preds, accuracy   

    def predict_freeman_edit_distance2(self, nb_imgs, img_to_predict):
        
        preds = []
        freemans = []

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = x_train[0:nb_imgs, : , :]
        y_train = y_train[0:nb_imgs]

        

        freemans, to_delete_indexes = freeman.freeman_representation(x_train)
        for index in to_delete_indexes:
            y_train = np.delete(y_train, (index), axis=0 )

        x_test_freemans, to_delete_indexes_x_test = freeman.freeman_representation(img_to_predict)
        
        

        for row in x_test_freemans:
            dists = []
            for freeman_img in freemans:
                dist = self.levenshtein_distance(freeman_img, row)
                dists.append(dist)
            print(dists)

            dists = np.array(dists)
            dists = dists.reshape((dists.shape[0], 1))
            print("dists shape: {}".format(dists.shape))
            y_train = y_train.reshape((y_train.shape[0], 1))
            print("y_train.shape: {}".format(y_train.shape))
            dists_with_classes = np.append(y_train, dists, axis=1)
            print(dists_with_classes.shape)

            sorted_dists = dists_with_classes[dists_with_classes[:, 1].argsort()]
            print(sorted_dists)
            print(sorted_dists)
            k_nearest_neighboors = sorted_dists[:self.k, :]
            print(k_nearest_neighboors)
            knn_classes = k_nearest_neighboors[:,0].astype(int)
            pred = np.bincount(knn_classes).argmax()
            preds.append(pred)
        print("preds : {}".format(np.array(preds)))
        # On compte le nombre de bonne prédictions        
        return preds[0]   


def main():
    knn = Knn(1)

    train_x = np.array([[0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]])
    train_y = np.array([3, 3, 3, 2 ])

    test_x = np.array([[1, 1, 1], [1, 0, 0]])
    test_y = np.array([2, 3])

    print(test_y)
    knn.predict_euclidian_dist(train_x, train_y, test_x, test_y)

def main2():
    knn = Knn(1)
    #print(knn.levenshtein_distance_DP(np.array([1, 3, 3]), np.array([1, 3, 3])))


    train_x = np.array([[0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 1]])
    train_y = np.array([3, 3, 3, 2 ])

    test_x = np.array([[1, 1, 1], [1, 0, 0]])
    test_y = np.array([2, 3])

    print(test_y)
    knn.predict_levenshtein_dist(train_x, train_y, test_x, test_y)


def main3():
    knn = Knn(3)
    knn.predict_freeman_edit_distance(4000, 4)




if __name__ == "__main__":  
    main3()