import numpy as np

# data
X = np.array([[1, 2], [1, 3], [1, 7]])
Y = np.array([2, 5, 6])

# theta values
W = np.zeros(2)
#W = np.ones(2)

print('X, Y, W')
print(X, Y, W)

# now coding gradient descent
epoch = 5000
precision = 0.0000001
alpha = 0.01
N = np.shape(X)[0]

#points = np.genfromtxt("simpledata.csv", delimiter=",")
#print(points)

for i in range(epoch):
    XdotW = np.dot(X, W)

    # experimenting alternative to dot product:
    # Comment from lr2.py: @ means matrix multiplication of arrays. If we want to use * for
    # multiplication we will have to convert all arrays to matrices
    XmW = X @ W
    XdotWMinusY = XdotW - Y
    G = np.dot(XdotWMinusY, X) / N
    if i < 5:
        print(i + 1,' ',G)
        print('XdotW')
        print(XdotW)
        print('XmW')
        print(XmW)
        print('XdotWMinusY')
        print(XdotWMinusY)
        print('X')
        print(X)
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
