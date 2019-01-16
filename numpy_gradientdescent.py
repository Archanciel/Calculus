import numpy as np

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
Y = data[:,-1:]
print('\nY extracted from data')
print(Y)
Y = Y.flatten() # algo not working if not flattening !
print('\nY flattened')
print(Y)

# starting theta values
W = np.zeros(2)

print('\nW')
print(W)

# now coding gradient descent

epoch = 5000
precision = 0.0000001
alpha = 0.01
N = np.shape(X)[0]

for i in range(epoch):
    XdotW = np.dot(X, W)
    XdotWMinusY = XdotW - Y
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
