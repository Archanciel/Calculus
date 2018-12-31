from sympy import *

# Source URL: https://firsttimeprogrammer.blogspot.com/2014/09/multivariable-gradient-descent.html
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

f = x ** 2 + y ** 2 - 2 * x * y
#f = x ** 2 + y ** 2

# First partial derivative with respect to x
fpx = f.diff(x)

# First partial derivative with respect to y
fpy = f.diff(y)

print('f: ')
pprint(f)

print('\nfpx: ')
pprint(fpx)

print('\nfpy: ')
pprint(fpy)

# Gradient
grad = [fpx, fpy]

# Data
theta_x = 5  # x
theta_y = 5  # y
alpha = .01
iterations = 0
check = 0
precision = 1 / 1000000
printData = True
maxIterations = 1000

while True:
    slope_x = N(fpx.subs(x, theta_x).subs(y, theta_y))
    descent_value_x = alpha * slope_x
    temptheta_x = theta_x - descent_value_x
    slope_y = N(fpy.subs(y, theta_y).subs(x, theta_x))
    descent_value_y = alpha * slope_y
    temptheta_y = theta_y - descent_value_y

    # If the number of iterations goes up too much, maybe theta (and/or theta1)
    # is diverging! Let's stop the loop and try to understand.
    iterations += 1
    if iterations > maxIterations:
        print("Too many iterations. Adjust alpha and make sure that the function is convex!")
        printData = False
        break

    # If the value of theta changes less of a certain amount, our goal is met.
    if abs(temptheta_x - theta_x) < precision and abs(temptheta_y - theta_y) < precision:
        print('precision {} reached'.format(precision))
        break

#    print('theta_x: ', theta_x, '\tslope_x', slope_x, '\tdescent_value_x', descent_value_x, '\ttheta_y: ', theta_y, '\tslope_y', slope_y, '\tdescent_value_y', descent_value_y)
#    print('theta_x: ', theta_x, '\tslope_x', slope_x, '\ttheta_y: ', theta_y, '\tslope_y', slope_y)
    print('(x, y) ({0:3.4f}, {1:3.4f}'.format(theta_x, theta_y), '\tslope_x', slope_x, '\tslope_y', slope_y)
    # Simultaneous update
    theta_x = temptheta_x
    theta_y = temptheta_y

if printData:
    print("The function " + str(f) + " converges to a minimum")
    print("Number of iterations:", iterations, sep=" ")
    print("theta (x0) =", temptheta_x, sep=" ")
    print("theta1 (y0) =", temptheta_y, sep=" ")

# Output
#
# The function x**2 - 2*x*y + y**2 converges to a minimum
# Number of iterations: 401
# theta (x0) = 525.000023717248
# theta1 (y0) = 524.999976282752