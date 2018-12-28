from sympy import *
import sys
import numpy as np
from matplotlib import pyplot as plt

x = Symbol('x')

# Function
y = x**2
# First derivative with respect to x
yprime = y.diff(x)
# Second derivative with respect to x
ypp = yprime.diff(x)

def plotFun():
    space = np.linspace(-5,5,100)
    data = np.array([N(y.subs(x,value)) for value in space])
    plt.plot(space, data)
    plt.show()

theta = 84
theta2 = 0
alpha = .01
iterations = 0
check = 0
precision = 1/10000
plot = False
iterationsMax = 10000
maxDivergence = 50

while True:
    theta2 = theta - alpha*N(yprime.subs(x,theta)).evalf()
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
            plot = False
            break
          
    # If the value of theta changes less that a certain
    # tolerance, we stop the program since theta has
    # converged to a value.
    if abs(theta - theta2) < precision:
        break
    
    theta = theta2


if plot:
    print("Number of iterations:",iterations,"value of theta:",theta2,sep=" ")
    plt.plot(theta,N(y.subs(x,theta)).evalf(),marker='o',color='r')
    plotFun