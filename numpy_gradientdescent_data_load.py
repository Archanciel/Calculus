import numpy as np
import matplotlib.pyplot as plt

# loading data
data = np.genfromtxt("simpledata.csv", delimiter=",")
print('data')
print(data)

# extracting X from data
X = data[:,:-1]
print('\nX extracted from data')
print(X)

# adding a column of 1's to the X matrix
X = np.insert(X,0,1,axis=1)
print('\nX after adding col of ones')
print(X)

# extracting Y from data
y = data[:, -1:]
print('\nY extracted from data')
print(y)
y = y.flatten() # algo not working if not flattening !
print('\ny flattened')
print(y)

# starting theta values
W = np.zeros(X.shape[1])

print('\nW')
print(W)

# now coding gradient descent

epoch = 5000
precision = 0.0000001
alpha = 0.01
N = np.shape(X)[0]

for i in range(epoch):
    XdotW = np.dot(X, W) # The way that numpy is programmed means that a 1D array, shape=(n,),
                         # is treated as either a column or a row vector based on the position
    # in a dot product.
    XdotWMinusY = XdotW - y
    G = np.dot(XdotWMinusY, X) / N
    if i < 5:
        print('iter ', i + 1)
        print('X')
        print(X)
        print('W')
        print(W)
        print('XdotW')
        print(XdotW)
        print('XdotWMinusY')
        print(XdotWMinusY)
        print(G)
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

#plt.scatter(X[:, 1].reshape(-1, 1), y, s = 500)

plt.scatter(X[:, 1], y, s=50) # removed call to reshape()

# axes = plt.gca()
# x_vals = np.array(axes.get_xlim())
# print('\nx_vals')
# print(x_vals)
# y_vals = W[0] + W[1] * x_vals # the line equation
# print('\ny_vals')
# print(y_vals)
# plt.plot(x_vals, y_vals, '-')

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
