#!/usr/bin/env python3

from keras.models import load_model
import cv2
import numpy as np
from PIL import ImageTk, Image, ImageDraw
import PIL
import tkinter as tk
from tkinter import *
import freeman
import nn
from keras.models import load_model
import tensorflow as tf



width = 500
height = 500
center = height//2
white = (255, 255, 255)
green = (0,128,0)

        
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=40)
    draw.line([x1, y1, x2, y2],fill="black",width=40)

def model2():
    n = nn.Neuralnetwork()
    n.fit(5)
    filename = "image.png"
    image1.save(filename)
    img=cv2.imread('image.png',0)
    img=cv2.bitwise_not(img)
    img=cv2.resize(img,(28,28))
    binary_img = freeman.convert_binary(img)
    print(binary_img)
    binary_img = binary_img.reshape(1, 28, 28)
    print(binary_img)
    preds = n.predict(binary_img)
    txt.insert(tk.INSERT,"{}\n".format(preds[0]))

def model():
    model=tf.keras.models.load_model('digit_recognition_optimized.h5')
    filename = "image.png"
    image1.save(filename)   
    img=cv2.imread('image.png',0)
    img=cv2.bitwise_not(img)
    img=cv2.resize(img,(28,28))
    img = img.reshape(1, 28, 28)
    preds = model.predict(img)
    preds = np.argmax(preds, axis=1)
    print(preds)
    txt.insert(tk.INSERT,"{}\n".format(preds[0]))

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
root.title('digit recognizer - Neural Network')
root.mainloop()