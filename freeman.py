 
#!/usr/bin/env python3

from keras.datasets import mnist
from matplotlib.pyplot import axis
import numpy as np

directions = [ 0,  1,  2,
               7,      3,
               6,  5,  4]
dir2idx = dict(zip(directions, range(len(directions))))

change_j =   [-1,  0,  1, # x or columns
              -1,      1,
              -1,  0,  1]

change_i =   [-1, -1, -1, # y or rows
               0,      0,
               1,  1,  1]

def binary(m):
    if m < 0.5 :
        return 0
    else:
        return 1

#Convertit les images de imgs en images binaires 
def convert_binary(imgs):
    binary_vec = np.vectorize(binary)
    binary_imgs = binary_vec(imgs)
    return binary_imgs


def freeman_chain_code(img):

    #On cherche le premier pixel plein de départ
    for i, row in enumerate(img):
        for j, value in enumerate(row):
            if value == 1:
                start_point = (i, j)
                print(start_point, value)
                break
        else:
            continue
        break
    
    border = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
        if img[new_point] != 0: # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break
    count = 0

    while curr_point != start_point:
        #figure direction to start search
        b_direction = (direction + 5) % 8 
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
            if img[new_point] != 0: # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        if count == 1000: break
        count += 1
    print(count)
    return chain  


#convert a whole dataset with freeman representation
def freeman_representation(x_train):
    freemans = []
    # on récupère les indexs des images dont on ne peut pas calculer la représentation de freeman car dessin sur les bords
    to_delete_indexes = []
    #On convertit en image binaire tout le dataset
    binary_x_train = convert_binary(x_train)

    for index, img in enumerate(binary_x_train):
        try:
            freeman = freeman_chain_code(img)
            freemans.append(freeman)
        except Exception as e:
            print(e)
            to_delete_indexes.append(index)
            print(img)

    return freemans, to_delete_indexes



def main():
    freemans = []

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    binary_x_train = convert_binary(x_train)
    binary_x_train = binary_x_train[0:200, :,:]
    y_train = y_train[0:200]
    freemans, to_delete_indexes = freeman_representation(binary_x_train)
    print(freemans)
    print(to_delete_indexes)
    print(y_train.shape)
    for index in to_delete_indexes:
        y_train = np.delete(y_train, (index), axis=0 )
    print(len(freemans))
    print(y_train.shape)
   


        
    
if __name__ == "__main__":  
    main()



