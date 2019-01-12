from sympy import *
import numpy as np
from matplotlib import pyplot as plt

# Source URL: https://firsttimeprogrammer.blogspot.com/2014/09/gradient-descent-algorithm.html?m=1
X_RANGE_RIGHT = 1
X_RANGE_LEFT = -3.2
x = Symbol('x')

# Function
#y = x**2
y = x**4 + 4*x**3 + 3*x**2
pprint(y)
# First derivative with respect to x
yprime = y.diff(x)
# Second derivative with respect to x
ypp = yprime.diff(x)

# space = np.linspace(-5, 5, 100)
# print(space)
# print([N(y.subs(x, value)) for value in space])

theta = X_RANGE_RIGHT
#theta = -1
# theta = -X_RANGE_RIGHT

def plotFun():
    space = np.linspace(X_RANGE_LEFT,X_RANGE_RIGHT,100)
    data = np.array([N(y.subs(x,value)) for value in space])
    plt.plot(space, data)
    plt.show()

theta2 = 0
alpha = .01
iterations = 0
check = 0
precision = 1/10000
plot = True
iterationsMax = 10000
maxDivergence = 50

# sizing the matplotlib window. This must be done before any plot operation
plt.rcParams["figure.figsize"] = (6, 10)  # no effect !!ü

while True:
    slope = N(yprime.subs(x, theta))
    descent_value = alpha * slope
    theta2 = theta - descent_value
    iterations += 1
    
    # If we make too much iterations our program
    # stops and we need to check it to be sure the
    # parameters are correct and it is working properly
    if iterations > iterationsMax:
        print("Too much iterations")
        break
      
    # Check if theta converges to a value or not
    # We allow a max of 50 divergences
    if theta < theta2:
        print("The value of theta is diverging")
        check += 1
        if check > maxDivergence:
            print("Too much iterations (%s), the value of theta is diverging"%maxDivergence)
            print("Please choose a smaller alpha and, or check that the function is indeed convex")
#            plot = False
            break
          
    # If the value of theta changes less that a certain
    # tolerance, we stop the program since theta has
    # converged to a value.
    if abs(theta - theta2) < precision:
        print('precision {} reached'.format(precision))
        break
    
    print('theta: ', theta, '\tslope', slope, '\tdescent_value', descent_value, '\ttheta2: ', theta2)

    if plot:
        plt.plot(theta, N(y.subs(x, theta)), marker='o', color='r')

    theta = theta2

if plot:
    plt.plot(theta,N(y.subs(x,theta)),marker='o',color='r')
    plotFun()

print("Number of iterations:",iterations,"value of theta:",theta2,sep=" ")
