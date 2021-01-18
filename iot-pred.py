'''
EL OBJETIVO DEL SCRIPT ES PODER PREDECIR CUANTO DEMORA UN CALEFON EN ALCANZAR LA TEMPERATURA ACONSEJADA
POR LOS DERMATOLOGOS: 38°C Y CUANTO DEMORA EN BAJAR A 15 ° C UNA VEZ DESACTIVADO EL CALENTADOR. 
SE TIENEN EN CUENTA LAS SIGUIENTES VARIABLES=
   1.  TEMPERATURA AMBIENTE(°C)
   2.  METROS CUBICOS DEL AMBIENTE.
   3.  TOMAMOS SEGUN LA POTENCIA PROMEDIO STANDARD QUE ES 2000 WATTS


PARA QUE SIRVE?
SABER CUANTO DEMORA PARA DE ESA MANERA ACTIVAR EL CALEFON VIA REMOTA EN EL MOMENTO PRECISO PARA UTILIZARLO
EN LA TEMPERATURA DESEADA SIN QUE SE ENFRIE. 

TODOS LOS DATOS SERIAN SUMINISTRADOS POR LOS DISPOSITIVOS IOT QUIENES LO CARGAN EN UNA BB.DD DE DONDE 
LO EXTRAERIA EL SCRIPT(parte del codigo a desarrollar)
'''


import numpy as np
import matplotlib.pyplot as plt
import random




#np.array=(temperatura ambiente/metros cubicos/)
X =np.array(([27,18],[15,18.],[20,18],[14.,12],[23,12],[30,12],[12,18],[10,18],[10,12],[12,18],[18,18],[11,12],[9,18],[8,12],[12,24],[19,18],[19,24],[31,24],[31,18],[9,24],[28,18],[28,30],[25,18],[27,12],[29,18],[36,18],[30,24],[11,24],[0,18],[2,12]),dtype=float)

#y=np.array(tiempo que tarda en llegar a 38° C, tempetatura optima según dermatologos, jamas debe pasar los 41 grados// potencia de calefon 2000 watts es el standard)
y = np.array(([10,8],[15,13],[12,10],[12,7],[8,12],[5,11],[9,10],[11,7],[9,10],[10,12],[9,13],[8,12],[10,11],[6,12],[5,13],[10,11],[6,10],[4,13],[7,12],[6,11],[9,10],[6,11],[9,12],[4,11],[9,10],[7,11],[8,10],[5,13],[4,11],[5,12]), dtype=float)

#datos que se requieren para predecir
xPredicted = np.array(([25,8]), dtype=float)

X = X/np.amax(X, axis=0)
xPredicted = xPredicted/np.amax(xPredicted, axis=0) 
#dividimos el valor por 10.000 para obtener un valor entre 0 y 1, la red trabaja en esos margenes. 
#Luego la prediccion la multiplicamos x 100.000 para obtener el resultado
y = y/10000

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 2
    self.outputSize = 2
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
    print ("DEMORA EN ALCANZAR LOS 38 °C Y EN APAGADO BAJAR A 15°C:   \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
for i in range(100): # cantidad de veces que se repite el entrenamiento
  print ("Datos de entrada: \n" + str(X))
  print ("Datos de salida: \n" + str(y))
  print ("Predicciones: \n" + str(NN.forward(X)))
  print ("Error: \n" + str(np.mean(np.square(y - NN.forward(X))))) 
  print ("\n")
  NN.train(X, y)

NN.saveWeights()
NN.predict()


