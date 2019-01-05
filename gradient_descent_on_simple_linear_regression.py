import numpy as np

data = np.genfromtxt("data.csv", delimiter=",")
print(data)
N = len(data)
epochs = 1000
alpha = 1 / epochs

# y = t0 + t1x
t0 = 1
t1 = 1
print("t0 {} t1 {}".format(t0, t1))

grad0 = 0 # partial derivative t0 of cost function J
grad1 = 0 # partial derivative t1 of cost function J

for i in range(epochs):
    for j in range(N):
        x = data[j, 0]
        y = data[j, 1]
        grad0 += t0 + t1 * x - y
        grad1 += x * (t0 + t1 * x - y)
        #print("x {} y {}".format(x, y))
    grad0 /= N
    grad1 /= N
    t0 = t0 - alpha * t0 * grad0
    t1 = t1 - alpha * t1 * grad1
    print("grad0 {0:1.4f} grad1 {1:1.4f} t0 {2:1.4f} t1 {3:1.4f}\n".format(grad0, grad1, t0, t1))
print('y = {0:1.4f} + {1:1.4f}x'.format(t0, t1))