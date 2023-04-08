import numpy as np
import matplotlib.pyplot as plt
from random import random

#Datos conocidos. Entradas y bias
matrizEntradas = np.array( [    [0,0,0,random()],
                                [0,0,1,random()],
                                [0,1,0,random()],
                                [0,1,1,random()],
                                [1,0,0,random()],
                                [1,0,1,random()],
                                [1,1,0,random()],
                                [1,1,1,random()]] )
salidasDeseada = np.array( [0,1,2,3,4,5,6,7] )

#Inicializacion de los pesos
w = np.array([random(), random(),random(),random()])

errorHistorico = []
error = 1
errorUmbral = 0.01
errorCuadratico = 0
factorAprendisaje = 0.2
errorCuadraticoMedio = 0
epocas = 0


while np.abs(error) > errorUmbral:
    #Iteramos dependiendo el numero de muestras (3 entradas y el umbral)
    for patron, entradas in enumerate(matrizEntradas):
        salidaModelo = (entradas*w).sum()
        errorLocal = (salidasDeseada[patron] - salidaModelo)
        w = w + factorAprendisaje * errorLocal * entradas
        errorCuadratico += errorLocal**2
    
    errorAux = errorCuadraticoMedio
    errorCuadraticoMedio = errorCuadratico/ len(matrizEntradas)
    error = errorCuadraticoMedio - errorAux
    errorHistorico.append(np.abs(error))

    epocas += 1


print("epocas: ", epocas)
print("pesos: ", w)


#Validacion

while True:

    x = input("Ingresar numero binario de tres digitos: ")
    x = np.array(list( map( lambda _: int(_), x)))
    print( round( (x*w[:-1]).sum() ) )