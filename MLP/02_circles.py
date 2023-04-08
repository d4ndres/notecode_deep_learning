import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from MLP import MLP

# Data
n = 500 
p = 2   
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

# Create and train

topology = [p, 4, 8, 1]
neural_n = MLP(topology)
neural_n.train( X, Y, epoch = 2000, min_loss = 0.01 ,lr = 0.05)


# Validation and plot

res = 50
_x0 = np.linspace(-1.5, 1.5, res)
_x1 = np.linspace(-1.5, 1.5, res)
_Y = np.zeros((res, res))

# Clash pretiction
for i0, x0 in enumerate(_x0):
    for i1, x1 in enumerate(_x1):
        _Y[i0, i1] = neural_n.prediction(np.array([[x0,x1]]) )[0][0]

# Plot top view
plt.figure(3, figsize=(15,5))
plt.subplot(1,2,1)
plt.pcolormesh(_x0, _x1, _Y, cmap = "viridis")
plt.axis("equal")

plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c = "skyblue" )
plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c = "salmon" )

plt.subplot(1,2,2)
plt.plot(range(len(neural_n.loss)), neural_n.loss)
plt.show()
