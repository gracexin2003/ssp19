#Grace Xin, Math/Pset 4, Problem 5
#Due July 3
import matplotlib.pyplot as plt

def euler(h, iterations, x, v):
    X, V = [x], [v]
    for i in range(iterations):
        newX = x+h*v
        newV = v-h*x
        X.append(newX)
        V.append(newV)
        x = newX
        v = newV
    return X, V

X, V = euler(0.5, 20, 1, 0) # what are the initial values of x and v?
#print(X)
#print(V)
#'''
for i in range(len(X)):
    plt.scatter(X[i], V[i], s=25)
#'''
#plt.scatter(X,V)
plt.plot(X,V)
plt.show()

