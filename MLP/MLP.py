import numpy as np

class Neural_layer():
    # La unidad del procesamiento es la neurona. Entonces se puede pensar como una objeto o un nivel superior como una capa de nuronas.
    # Una capa de neuronas artificial debe tener: número de conexiones, el número de neuronas, la función de activación.
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1     , n_neur) * 2 - 1 #ballas # Declaracion normalizada aleatorios de  -1 a 1 
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1 #tantas conexiones como neuronas


class MLP():
    # Funcion de activacion por defecto. Tupla que contiene la funcion sigmoidal y su derivada
    sigm = (lambda x: 1 / ( 1 + np.e ** (-x)), lambda x: x * (1 - x))

    # Calculo de error o coste. Tupla que contiene el error cuadratico medio y su derivada 
    l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2), lambda Yp, Yr: (Yp - Yr))

    # Historico del desenso del error
    loss = []

    # Constructor que crea la estructura de la red neuronal. (Neural net)
    def __init__( self, topology, act_f = None ):
        self.nn = []
        act_f = act_f if act_f else self.sigm

        for idx, value in enumerate(topology[:-1]): # podria quitar value y remplazar enumerate por range
            self.nn.append(Neural_layer(topology[idx], topology[idx+1], act_f))


    # Entrenamiento: Conjunto de entradas, conjunto de salidas, ratio de aprendisaje, ¿lo entrenamos o predecimo un resultado?
    def train( self, X, Y, epoch = 2000, min_loss = 0.01, lr = 0.5, train = True):

        # Guardaremos el valor de la suma ponderada, el valor de activacion. para cada capa.
        # Por lo que estamso guarda los pares de z y a [(z0,a0),(z1,a1),...]  
        out=[(None,X)] 

        #1. Propagación hacia adelante. Forwared pass. predicción no entrenada.
        for idx, layer in enumerate( self.nn ):

            z = out[-1][1] @ self.nn[idx].W + self.nn[idx].b
            a = self.nn[idx].act_f[0](z)
            out.append((z,a))


        #2. Backward pad.
        if train:
            delta=[]
        
            for idx in reversed(range(0,len(self.nn))):
                z=out[idx+1][0]
                a=out[idx+1][1]
        
                #Algoritmo delta de última capa
                if idx == len(self.nn) - 1:
                    #derivada de la funcion de coste * derivada de la funcion de activacion de la ultima capa
                    delta.insert(0, self.l2_cost[1](a, Y) * self.nn[idx].act_f[1](a))
                
                #Algorimo delta respecto a la capa previa
                else:
                    #(Valor del delta anterior @ vector de pesos W) * derivada de la funcion de activacion de la capa
                    delta.insert(0, delta[0] @ aux_w.T * self.nn[idx].act_f[1](a))


                # EL vector de pesos se guardado en un aux. Debido a su pos modificación
                aux_w = self.nn[idx].W

                #3. Gradient Descent: 
                # Ajuste del valor del vector de ballas y el valor la matriz de pesos.
                
                # Se restara el parametro b de la capa idx respecto al coste.
                # El coste: es la deribada parcial en funcion de delta. 
                    # la deribada parcial es: delta * lr)
                    # delta tiene la dimension de n numero de muestras. Por lo que se usa el valor medio. asi poder operar 1 a 1.

                self.nn[idx].b = self.nn[idx].b - np.mean(delta[0], axis=0, keepdims=True) * lr
                

                # Se restara los pesos de la capa idx respecto a la multiplicacion matricial. 
                # de la salida de la funcion de activacion de la capa acnterio @ ultimo delta calculado * lr

                self.nn[idx].W = self.nn[idx].W - out[idx][1].T @ delta[0] * lr
            
        # Finalmente retornamos la salida de la funcion de activacion de la ultima capa
        return out[-1][1]
