# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 02:31:46 2021

@author: Daniel Paez
"""


import numpy as np
import matplotlib.pyplot as plt
import random
'''
casos1=[ 3558], [5274],  [7112],  [6994],  [5303],  [3610], [3216],[3278],[5201],
         [6899]
fallecidos1=[98],[62],[175],[209],[213],[121],[118],[138],[120],[207]
       
print(casos1)
print(fallecidos1)
'''

# X = (metros cuadrados, baños,habitaciones,pisos), 
X = np.array(([ 2000], [2002],  [1976],  [1945],  [2011],  [1988], [2020],[1982],[2000],[1933]), dtype=float)
#precio de la casa
y = np.array(([1],[1],[0],[0],[1],[0],[1],[0],[1],[0]), dtype=float)
#datos que se requieren para predecir
xPredicted = np.array(([2020]), dtype=float)

X = X/np.amax(X, axis=0)
xPredicted = xPredicted/np.amax(xPredicted, axis=0) 
#dividimos el valor por 100.000 para obtener un valor entre 0 y 1, la red trabaja en esos margenes. 
#Luego la prediccion la multiplicamos x 100.000 para obtener el resultado
y = y/10000

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 1
    self.outputSize = 1
    self.hiddenSize = 2

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer 
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # funcion de activacion
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    #funcion de activacion.
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivada de Sigmoide
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error de salida
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print ("Prediccion: ")
    print ("Input (scaled): \n" + str(xPredicted))
    print ("Dato de salida: \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
for i in range(20000): # cantidad de veces que se repite el entrenamiento
  print ("Datos de entrada: \n" + str(X))
  print ("Datos de salida: \n" + str(y))
  print ("Predicciones: \n" + str(NN.forward(X)))
  print ("Error: \n" + str(np.mean(np.square(y - NN.forward(X))))) 
  print ("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()


'''
16/01/2021 0:30
Datos de entrada:
[[0.50028121 0.49226442]
 [0.74156355 0.74120956]
 [1.         1.        ]
 [0.98340832 0.98312236]
 [0.74564117 0.74542897]
 [0.5075928  0.50773558]
 [0.45219348 0.45147679]
 [0.46091114 0.45991561]
 [0.73129921 0.73136428]
 [0.97005062 0.96905767]]
Datos de salida:
[[0.0098]
 [0.0062]
 [0.0175]
 [0.0209]
 [0.0213]
 [0.0121]
 [0.0118]
 [0.0138]
 [0.012 ]
 [0.0207]]
Predicciones:
[[0.01653406]
 [0.01278285]
 [0.01118906]
 [0.01125815]
 [0.01274394]
 [0.01620865]
 [0.0175992 ]
 [0.01736879]
 [0.01287501]
 [0.01131968]]
Error:
4.4668482527201104e-05


Prediccion:
Input (scaled):
[1.  0.1]
Dato de salida:
[0.02429619] '''



'''
16/01/2021 1:30
Datos de entrada:
[[0.50028121 0.49226442]
 [0.74156355 0.74120956]
 [1.         1.        ]
 [0.98340832 0.98312236]
 [0.74564117 0.74542897]
 [0.5075928  0.50773558]
 [0.45219348 0.45147679]
 [0.46091114 0.45991561]
 [0.73129921 0.73136428]
 [0.97005062 0.96905767]]
Datos de salida:
[[0.0098]
 [0.0062]
 [0.0175]
 [0.0209]
 [0.0213]
 [0.0121]
 [0.0118]
 [0.0138]
 [0.012 ]
 [0.0207]]
Predicciones:
[[0.01512391]
 [0.01412776]
 [0.0149219 ]
 [0.01483647]
 [0.01412816]
 [0.01494789]
 [0.01549398]
 [0.01540345]
 [0.01412306]
 [0.01477248]]
Error:
2.5001144873351613e-05  


Prediccion:
Input (scaled):
[1.  0.1]
Dato de salida:
[0.02292613] 


16/01/2021  3:30
Datos de entrada:
[[0.50028121 0.49226442]
 [0.74156355 0.74120956]
 [1.         1.        ]
 [0.98340832 0.98312236]
 [0.74564117 0.74542897]
 [0.5075928  0.50773558]
 [0.45219348 0.45147679]
 [0.46091114 0.45991561]
 [0.73129921 0.73136428]
 [0.97005062 0.96905767]]
Datos de salida:
[[0.0098]
 [0.0062]
 [0.0175]
 [0.0209]
 [0.0213]
 [0.0121]
 [0.0118]
 [0.0138]
 [0.012 ]
 [0.0207]]
Predicciones:
[[0.0116572 ]
 [0.01469745]
 [0.01901126]
 [0.01872539]
 [0.01476194]
 [0.01176861]
 [0.01137286]
 [0.01142195]
 [0.01454309]
 [0.01849054]]
Error:
1.4271139391757991e-05


Prediccion:
Input (scaled):
[1.  0.1]
Dato de salida:
[0.01098971] 


16/01/2021  5:30
Datos de entrada:
[[0.50028121 0.49226442]
 [0.74156355 0.74120956]
 [1.         1.        ]
 [0.98340832 0.98312236]
 [0.74564117 0.74542897]
 [0.5075928  0.50773558]
 [0.45219348 0.45147679]
 [0.46091114 0.45991561]
 [0.73129921 0.73136428]
 [0.97005062 0.96905767]]
Datos de salida:
[[0.0098]
 [0.0062]
 [0.0175]
 [0.0209]
 [0.0213]
 [0.0121]
 [0.0118]
 [0.0138]
 [0.012 ]
 [0.0207]]
Predicciones:
[[0.01160805]
 [0.01462613]
 [0.01912991]
 [0.01882261]
 [0.01469185]
 [0.01172591]
 [0.01133676]
 [0.01138528]
 [0.01447118]
 [0.01856944]]
Error:
1.4173988188856095e-05


Prediccion:
Input (scaled):
[1.  0.1]
Dato de salida:
[0.00860058]'''