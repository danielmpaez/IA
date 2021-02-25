#IMPORTACION DE LIBRERIAS
import cv2
import numpy as np
import speech_recognition as sr
import subprocess as sub
import pyautogui as auto
import pyttsx3 as voz
from time import sleep
import msvcrt
import os 


os.system ("cls") #LIMPIAMOS PANTALLA

#CPARAMETRO DE SINTETIZADOR DE VOZ
engine = voz.init()
engine.setProperty('rate', 140)
engine.setProperty('voice' [1], 'spanish')
engine.setProperty('gender', 'male')
#engine.setProperty ('voice', voices [0] .id) #cambiar índice, cambiar voces. o para hombre
engine.setProperty('volume', 7)


#FUNCION DE VOZ
def saySomething(somethingToSay):
    engine.say(somethingToSay)
    engine.runAndWait()

saySomething("iniciando script de comparación de placas     ")

#CARGAMOS IMAGENES A COMPARAR,ESTO PUEDE HACERSE CON LA CAMARA EN VIVO
saySomething("cargando imágenes")
original = cv2.imread("1.jpeg")
image_to_compare = cv2.imread("3.jpg")
saySomething("impresion de pixeles de cada imagen")
print("pixeles original")
print(original)
print("pixeles image_to_compare")
print(image_to_compare)

# CHEQUEAMOS COINCIDENCIAS DE TAMAÑO
if original.shape == image_to_compare.shape:
    print('Las imagenes tiene el mismo tamaño y canal')
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)
    print(cv2.countNonZero(b))
    if (cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0):
        print('Las imagenes son completamente iguales')
    else: 
        print('Las imagenes no son iguales')

# CHEQUEMOS IGUALDAD
shift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = shift.detectAndCompute(original, None)
kp_2, desc_2 = shift.detectAndCompute(image_to_compare, None)
#CONTAMOS PIXELES EN 1
print("Keypoints 1st image", str(len(kp_1)))
print("Keypoints 2st image", str(len(kp_2)))

index_params = dict(algorithm=0, trees=5)
search_params = dict()

saySomething("realizando proceso de comparación por resta de pixeles")

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
for m, n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)

print("GOOD matches",len(good_points))

#IMPRIMIMOS PIXELES DIFERENTES
diferencia = cv2.subtract(original,image_to_compare)
print(diferencia)
#FUNCION DE COMPARACION CON IMPRESION DE RESULTADO
def compara(original,image_to_compare):
	diferencia =cv2.subtract(original,image_to_compare)
	if not np.any(diferencia):
		print("las imagenes son iguales")
        #saySomething("las imágenes son iguales")
	else:
        
	    print("las imagenes son distintas")
        
        
        

compara(original,image_to_compare)

# Estas nuevas lineas son para crear el archivo y pintar los matches de las imagenes
result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
cv2.imshow("Resultado", cv2.resize(result, None, fx = 0.4, fy=0.4))
cv2.imwrite("Feature_matching.jpg", result)
cv2.imwrite("im_diferencia.png",diferencia)

cv2.imshow("Original", original)
cv2.imshow("Imagen a comparar", image_to_compare)
cv2.imshow("diferencia",diferencia)
cv2.waitKey(0)
cv2.destroyAllWindows()