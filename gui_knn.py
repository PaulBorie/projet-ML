#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import ImageTk, Image, ImageDraw
import PIL
import tkinter as tk
from tkinter import *
import freeman
import knn
import naivebayes

width = 500
height = 500
center = height//2
white = (255, 255, 255)
green = (0,128,0)

        
def paint(event):
    x1, y1 = (event.x - 15), (event.y - 15)
    x2, y2 = (event.x + 15), (event.y + 15)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=40)
    draw.line([x1, y1, x2, y2],fill="black",width=40)

def model():
    k_nn = knn.Knn(10)
    filename = "image.png"
    image1.save(filename)
    img=cv2.imread('image.png',0)
    img=cv2.bitwise_not(img)
    img=cv2.resize(img,(28,28))
    binary_img = freeman.convert_binary(img)
    print(binary_img)
    binary_img = binary_img.reshape(1, 784)
    preds = k_nn.mnist_euclidian_dist(binary_img)
    txt.insert(tk.INSERT,"{}\n".format(preds))


def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)

root = Tk()

root.resizable(0,0)
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

txt=tk.Text(root,bd=3,exportselection=0,bg='WHITE',font='Helvetica',
            padx=10,pady=10,height=5,width=20)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

btnModel=Button(text="Predict",command=model)
btnClear=Button(text="clear",command=clear)
btnModel.pack()
btnClear.pack()
txt.pack()
root.title('Knn : binary image + euclidian distance')
root.mainloop()