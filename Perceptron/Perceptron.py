import numpy as np
import matplotlib.pyplot as plt


class Perceptron():
    
    w = []
    b = 0

    def __init__(self, description = ""):
        self.description = description

    def __repr__(self) -> str:
        return f"Perceptron({self.description})"
    
    def hardlim(self, x) -> int:
        return 1 if x >= 0 else 0

    def evalError(self, error ) -> int:
        return np.absolute( error ).sum()

    def train( self, matrizDeEntradas, salidasDeseadas, epochMax = 10):
        self.X = matrizDeEntradas = np.array(matrizDeEntradas)
        self.Y = salidasDeseadas = np.array(salidasDeseadas)
        
        error = np.ones( salidasDeseadas.size )
        self.w = np.random.rand( matrizDeEntradas.shape[-1], 1)
        self.b = np.random.rand()
        self.epoch = 0

        while self.evalError( error ):
            for patronIndex, patron in enumerate( matrizDeEntradas ):
                salidaCalculada = self.hardlim( patron@self.w + self.b )
                error[patronIndex] = salidasDeseadas[patronIndex] - salidaCalculada
                self.w = self.w + error[patronIndex]*patron.reshape( len(patron) ,1)
                self.b = self.b + error[patronIndex]

            self.epoch += 1


            if self.epoch == epochMax:
                print("Dimensiones insuficientes")
                return self.epoch

        print("Solucion encontrada")
        return self.epoch

    def graph( self ):
        if ( self.epoch > 0 ):
            plt.scatter( self.X[self.Y == 0,0] , self.X[self.Y == 0,1], c="blue")
            plt.scatter( self.X[self.Y == 1,0] , self.X[self.Y == 1,1], c="salmon")
            p1 = np.linspace( self.X.min() - 1, self.X.max() + 1, 10 )
            p2 =  -(self.w[0]/self.w[1])*p1 -self.b/self.w[1];
            plt.plot(p1,p2)
            plt.axis('equal')
            plt.xlim([self.X.min() - 1, self.X.max() + 1])
            plt.ylim([self.X.min() - 1, self.X.max() + 1])
            plt.grid()
            plt.show()

