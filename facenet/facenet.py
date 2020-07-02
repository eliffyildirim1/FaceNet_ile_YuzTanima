# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:29:25 2020

@author: Elif
"""
from deepface.basemodels import Facenet
from deepface.commons import functions
import os
import cv2
import numpy as np

model = Facenet.loadModel()

input_shape = model.layers[0].input_shape[1:3]

print("model input shape: ", model.layers[0].input_shape[1:])
print("model output shape: ", model.layers[-1].input_shape[-1])

#img12.jpg
image2 = input("Enter path for the image:")

#belirlenen path den fotoyu alır. Tüm dizinlerin yolu alta yazılır, kod çalışınca sadece resim adı ve uzantısı yazılır
img2 = cv2.imread("Yolu veriniz" + image2) 

#foto algılama
img2 = functions.detectFace(img2, input_shape)

#dizinde bulunan dosya isimlerini listeye alır
persons = os.listdir("yolu veriniz")

for person in persons:
    #foto algılama
    img1 = functions.detectFace("yolu veriniz" + person, input_shape)
	
    #vektör görüntülerini çıkarır
    img1_representation = model.predict(img1)[0,:]

    #.jpg kısmını temizler
    name = person.replace(".jpg", "")
	
    #vektör görüntülerini çıkarır
    img2_representation = model.predict(img2)[0,:]

    #görüntüler arasındaki mesafeyi hesaplar, kare değerini bulur
    distance_vector = np.square(img1_representation - img2_representation)

    #denge mesafesi
    distance = np.sqrt(distance_vector.sum())

    if distance < 10:
	    print(f"This person is name \"{name}\".")
	    break

if distance > 10:
	print("The person unknown!")