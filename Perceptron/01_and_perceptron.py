from Perceptron import Perceptron

if __name__ == "__main__":
    entradas = [[0,0],[0,1],[1,0],[1,1]]
    salidas = [0,0,0,1]

    neurona = Perceptron()
    neurona.train( entradas, salidas)
    neurona.graph()