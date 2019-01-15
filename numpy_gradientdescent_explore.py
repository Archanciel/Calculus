import numpy as np

# data
X = np.array([[1, 2], [1, 3], [1, 7]])
Y = np.array([2, 5, 6])

# theta values
W = np.zeros(2)
#W = np.ones(2)
print('X, Y, W')
print(X, Y, W)

N = np.shape(X)[0]
print('\nN')
print(N)

data = np.genfromtxt("simpledata.csv", delimiter=",")
print('\ndata')
print(data)

oneColumn = np.ones((N, 1))
print('\noneColumn')
print(oneColumn)

#data[:,0:1] = oneColumn
data = np.insert(data,0,1,axis=1)
print('\ndata mod')
print(data)

#Y = data[:,-1:]
print('\ndataY')
print(Y)

lstColIdx = np.shape(data)[1] - 1
data = np.delete(data,lstColIdx,axis=1)
print('\ndata mod2')
print(data)


# cost function
XWdot = np.dot(X, W)
print('\nXWdot')
print(XWdot)

XWdotMinusY = XWdot - Y
print('\nXWdotMinusY')
print(XWdotMinusY)

XWdotMinusYsquared = XWdotMinusY ** 2
print('\nXWdotMinusYsquared')
print(XWdotMinusYsquared)

J = XWdotMinusYsquared.sum() / (2 * N)
print('\nJ')
print(J)

G = np.dot(XWdotMinusY, X) / N
print('Gradients')
print(G)

