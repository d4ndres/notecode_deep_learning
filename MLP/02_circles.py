import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from MLP import MLP

n = 500 
p = 2   
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

topology = [p, 4, 8, 1]
neural_n = MLP(topology)
loss = []
epoc = 2000
min_loss = 0.01

for i in range(epoc):
    pY = neural_n.train( X, Y,  lr = 0.05)

    if i % 25 == 0:
        loss.append(neural_n.l2_cost[0](pY,Y))

    if loss[-1] < min_loss:
        print("Numero de epocas " + str(i))
        break


res = 50
_x0 = np.linspace(-1.5, 1.5, res)
_x1 = np.linspace(-1.5, 1.5, res)

_Y = np.zeros((res, res))

#Enfrentamiento predictivo de la red neuronal
for i0, x0 in enumerate(_x0):
    for i1, x1 in enumerate(_x1):
        _Y[i0, i1] = neural_n.train(np.array([[x0,x1]]), Y,train=False)[0][0]

#Visualizamos los resultados del entrenamiento.
plt.figure(3, figsize=(15,5))
plt.subplot(1,2,1)
plt.pcolormesh(_x0, _x1, _Y, cmap = "viridis")
plt.axis("equal")

plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c = "skyblue" )
plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c = "salmon" )

plt.subplot(1,2,2)
plt.plot(range(len(loss)), loss)
plt.show()
