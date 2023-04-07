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

pY = None
for i in range(epocas):
    pY = nn.train( x, y )
    loss.append(nn.l2_cost[0](pY,y))

    epocasVividas += 1

    if minError > loss[-1]:
        break


print(f"epocasVividas: {epocas}\t Error: {loss[-1]}")
for i in range(len(y)):
    print(x[i], ": ", pY[i])

plt.plot(loss)
plt.show()
