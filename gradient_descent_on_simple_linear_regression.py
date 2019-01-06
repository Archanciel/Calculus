import numpy as np

'''
OUTPUT
t0: 0 t1: 0
Max precision of 1e-07 reached at epoch 3315)
y = 1.76187 + 0.64286x   J 12.99998

t0: 10 t1: 10
Max precision of 1e-07 reached at epoch 3732)
y = 1.76194 + 0.64285x   J 13.00002

t0: -10 t1: -10
Max precision of 1e-07 reached at epoch 3858)
y = 1.76187 + 0.64286x   J 12.99998
'''

data = np.genfromtxt("simpledata.csv", delimiter=",")
print(data)
N = len(data)
epochs = 5000
alpha = 0.01
precision = 0.0000001

def costFunction(data, t0, t1):
    cost = 0
    for i in range(N):
        cost += t0 + t1 * data[i, 0]

    return cost

def gradientDescent(t0, t1):
    # y = t0 + t1x

    J = 0
    grad0 = 0  # partial derivative t0 of cost function J
    grad1 = 0  # partial derivative t1 of cost function J

    print("t0: {} t1: {}".format(t0, t1))

    for i in range(epochs):
        #    printGrad0 = ''
        #    printGrad1 = ''

        for j in range(N):
            x = data[j, 0]
            y = data[j, 1]
            grad0 += t0 + t1 * x - y
            grad1 += x * (t0 + t1 * x - y)
        #        printGrad0 += '({} + {} * {} - {}) + '.format(t0, t1, x, y)
        #        printGrad1 += '({} + {} * {} - {}) * {} + '.format(t0, t1, x, y, x)
        grad0 /= N
        grad1 /= N
        # grad0 *= 2/N
        # grad1 *= 2/N
        tNew0 = t0 - alpha * grad0
        tNew1 = t1 - alpha * grad1

        if abs(t0 - tNew0) < precision and abs(t1 - tNew1) < precision:
            print('Max precision of {} reached at epoch {})'.format(precision, i))
            break

        t0 = tNew0
        t1 = tNew1

    J = costFunction(data, t0, t1)
    #    print('grad0: {0} = {1:1.12f}'.format(printGrad0, grad0))
    #    print('grad1: {0} = {1:1.12f}'.format(printGrad1, grad1))
    #    print("grad0 {0:1.10f} grad1 {1:1.10f} t0 {2:1.5f} t1 {3:1.5f} J {4:3.5f}\n".format(grad0, grad1, t0, t1, J))
    return t0, t1, J

t0, t1, J = gradientDescent(t0=0, t1=0)
print('y = {0:1.5f} + {1:1.5f}x   J {2:3.5f}\n'.format(t0, t1, J))

t0, t1, J = gradientDescent(t0=10, t1=10)
print('y = {0:1.5f} + {1:1.5f}x   J {2:3.5f}\n'.format(t0, t1, J))

t0, t1, J = gradientDescent(t0=-10, t1=-10)
print('y = {0:1.5f} + {1:1.5f}x   J {2:3.5f}\n'.format(t0, t1, J))