import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP

x = np.array([
[0,0], 
[0,1], 
[1,0], 
[1,1] ])
y = np.array([[0],[1],[1],[0]])
p = 2
nn = MLP( [p, 4, 2, 1])
loss = []
epocas = 5000
minError = 0.005
epocasVividas = 0

nn.train( x, y, epocas, minError )


for i in range(len(y)):
    print(x[i], ": ", nn.prediction( x[i]))

plt.plot(nn.loss)
plt.show()
