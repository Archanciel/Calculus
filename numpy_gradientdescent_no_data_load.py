import numpy as np
import matplotlib.pyplot as plt

# data
X = np.array([[1, 2], [1, 3], [1, 7]])
y = np.array([2, 5, 6])

# theta values
W = np.zeros(X.shape[1])
#W = np.ones(X.shape[1])

print('X, y, W')
print(X, y, W)

# now coding gradient descent
epoch = 5000
precision = 0.0000001
alpha = 0.01
N = np.shape(X)[0]

for i in range(epoch):
    XdotW = np.dot(X, W) # The way that numpy is programmed means that a 1D array, shape=(n,),
                         # is treated as either a column or a row vector based on the position
    XdotWMinusY = XdotW - y
    G = np.dot(XdotWMinusY, X) / N
    if i < 5:
        print(i + 1,' ',G)
        # print('XdotW')
        # print(XdotW)
        # print('XdotWMinusY')
        # print(XdotWMinusY)
        # print('X')
        # print(X)
#    print('\nold W')
#    print(W)
    newW = W - alpha * G

    diff = abs(newW - W)
#    diff = newW - W
#    print(diff)
    # print(np.all(np.less(diff,precision)))
    if np.all(np.less(diff,precision)):
        print('Max precision of {} reached after {} iteration'.format(precision, i + 1))
        print('Final W')
        print(newW)
        XdotWMinusYsquared = XdotWMinusY ** 2
        J = XdotWMinusYsquared.sum() / N
        print('\nJ')
        print(J)
        break

    W = newW

plt.scatter(X[:,1], y, s=50)
axes = plt.gca()
xVals = np.array(axes.get_xlim())
print('\nxVals')
print(xVals)

print('\nW')
print(W)
yVals = W[0] + W[1] * xVals # element wise addition and multiplication

print('\nW[1] * xVals')
print(W[1] * xVals)

print('\nyVals')
print(yVals)
plt.plot(xVals, yVals)
plt.show()