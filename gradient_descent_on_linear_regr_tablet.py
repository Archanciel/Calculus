from numpy import *

# Article: https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
# Source: https://github.com/mattnedrich/GradientDescentExample

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate, gradTimesTwo=True):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(y - ((m_current * x) + b_current))
        m_gradient += -x * (y - ((m_current * x) + b_current))
    if gradTimesTwo:
        b_gradient = b_gradient * 2 / N
        m_gradient = m_gradient * 2 / N
    else:
        b_gradient /= N
        m_gradient /= N
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, precision, num_iterations, use_precision=False, gradTimesTwo=True, doPrint = False):
    b = starting_b
    m = starting_m
    old_b = starting_b
    old_m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate, gradTimesTwo)
        if use_precision:
            print(old_b, b, old_m, m)
            if abs(old_b - b) < precision and abs(old_m - m) < precision:
                print('Max precision ({}) reached after {} iterations'.format(precision, i + 1))
                return b, m, i + 1
        if doPrint:
            print('{0:4} m {1:1.4f} b {2:1.4f}'.format(i + 1, m, b))
            print(old_b, b, old_m, m)
        old_b = b
        old_m = m

    return [b, m, i + 1]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    precision = 0.0001
    use_precision = False
    gradTimesTwo = True

    if use_precision:
        learning_rate = 0.0001
        num_iterations = 1000
    else:
        if gradTimesTwo:
            learning_rate = 0.0001
            num_iterations = 1000
        else:
            # Pour obtenir les mêmes valeurs pour b et m,
            # si le gradient n'est pas multiplié par 2, soit on double le learning rate
            # soit on double le nombre d'itérations,
            learning_rate = 0.0002
            num_iterations = 1000
    #        num_iterations = 2000
    #        learning_rate = 0.0001

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m, num_iterations] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, precision, num_iterations, use_precision, gradTimesTwo, doPrint=True)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()
