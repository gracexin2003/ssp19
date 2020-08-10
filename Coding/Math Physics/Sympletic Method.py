#Grace Xin, Math/Pset 4, Problem 6
#Due July 3
import matplotlib.pyplot as plt

def sympletic(h, iterations, x, v):
    X, V = [x], [v]
    for i in range(iterations):
        newX = (1/(1+(h**2)/4))*(x*(1-(h**2)/4)+h*v)
        newV = (1/(1+(h**2)/4))*(v*(1-(h**2)/4)-h*x)
        X.append(newX)
        V.append(newV)
        x = newX
        v = newV
    return X, V

X, V = sympletic(9, 50, 1,2) # what are the inital values of x and v?
#print(X)
#print(V)
'''
for i in range(len(X)):
    plt.scatter(X[i], V[i], s=25)
'''
plt.scatter(X,V)
plt.plot(X,V)
plt.show()
