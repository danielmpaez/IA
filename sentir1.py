# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 00:42:14 2021

@author: 54112
"""

from textblob import TextBlob
#Positiva y subjetiva


t=TextBlob("Ahora, para la gente, el país que les dejo es un país muy cómodo, es un país con 6,9 por ciento de desocupación, es un país con 6 millones de jubilados, es un país con el salario más alto de Latinoamérica, es un país con la jubilación más alta de Latinoamérica, es un país con la mayor inclusión previsional de que se tenga memoria, es un país con mayor nivel de porcentaje industrial en su Producto Bruto Interno, es un país donde se respetan los derechos humanos, es un país donde se respeta la división de lo que lo que es la Constitución, es un país donde el gobierno le ha dado más que nunca el mayor presupuesto al Poder Judicial")
ten=t.translate(to="en")
print(ten)

print(ten.sentiment,"/n")



