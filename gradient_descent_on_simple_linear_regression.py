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
epochs = 500000
alpha = 0.01 # value for simpledata.csv
#alpha = 0.0001 # value for data.csv
precision = 0.0000001

def costFunction(data, t0, t1):
    cost = 0
    for i in range(N):
        x = data[i, 0]
        y = data[i, 1]
        cost += ((t0 + t1 * x) - y) ** 2

    return cost / len(data)

def gradientDescent(t0, t1, doPrint=False):
    # y = t0 + t1x

    J = 0

    print("t0: {} t1: {}".format(t0, t1))

    for i in range(epochs):
        grad0 = 0  # partial derivative t0 of cost function J
        grad1 = 0  # partial derivative t1 of cost function J
        printGrad0 = ''
        printGrad1 = ''

        for j in range(N):
            x = data[j, 0]
            y = data[j, 1]
            grad0 += t0 + t1 * x - y
            grad1 += x * (t0 + t1 * x - y)
            if doPrint and i < 5:
                printGrad0 += '({0:1.3f} + {1:1.3f} * {2:1.3f} - {3:1.1f}) + '.format(t0, t1, x, y)
                printGrad1 += '({0:1.3f} * ({1:1.3f} + {2:1.3f} * {3:1.3f} - {4:1.1f})) + '.format(x, t0, t1, x, y, x)
        grad0 /= N
        grad1 /= N
        if doPrint and i < 5:
            print(i + 1, ': ' + printGrad0[:-2] + '= {0:1.5f} / {1} = {2:1.5f}'.format(grad0 * N, N, grad0))
            print(i + 1, ': ' + printGrad1[:-2] + '= {0:1.5f} / {1} = {2:1.5f}'.format(grad1 * N, N, grad1))
        # grad0 *= 2/N
        # grad1 *= 2/N
        tNew0 = t0 - alpha * grad0
        tNew1 = t1 - alpha * grad1

        if abs(t0 - tNew0) < precision and abs(t1 - tNew1) < precision:
            print('Max precision of {} reached at epoch {})'.format(precision, i + 1))
            break

        t0 = tNew0
        t1 = tNew1

    J = costFunction(data, t0, t1)
    #    print('grad0: {0} = {1:1.12f}'.format(printGrad0, grad0))
    #    print('grad1: {0} = {1:1.12f}'.format(printGrad1, grad1))
    #    print("grad0 {0:1.10f} grad1 {1:1.10f} t0 {2:1.7f} t1 {3:1.7f} J {4:3.5f}\n".format(grad0, grad1, t0, t1, J))
    return t0, t1, J

t0, t1, J = gradientDescent(t0=0, t1=0, doPrint=True)
print('y = {0:1.7f} + {1:1.7f}x   J {2:3.5f}\n'.format(t0, t1, J))

# t0, t1, J = gradientDescent(t0=10, t1=10)
# print('y = {0:1.7f} + {1:1.7f}x   J {2:3.5f}\n'.format(t0, t1, J))
#
# t0, t1, J = gradientDescent(t0=-10, t1=-10)
# print('y = {0:1.7f} + {1:1.7f}x   J {2:3.5f}\n'.format(t0, t1, J))
