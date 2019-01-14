import numpy as np

# data
X = np.array([[1, 2], [1, 3], [1, 7]])
Y = np.array([2, 5, 6])

# theta values
W = np.zeros(2)
#W = np.ones(2)
print('X, Y, W')
print(X, Y, W)


# cost function
N = np.shape(W)[0]
print('\nN')
print(N)

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
