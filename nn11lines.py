import numpy as np

# Code tiré de l'article ttps://iamtrask.github.io/2015/07/12/basic-python-network/

# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
        # xExp = np.exp(x)
        # return xExp / ((xExp + 1) ** 2)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1
print('syn0 ', syn0)
for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

#    print(iter)
#    print('l1 ',l1)

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)
#    print('l1_delta ', l1_delta)
#    input('return to continue')

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l1)