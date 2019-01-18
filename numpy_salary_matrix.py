import numpy as np
# salary in ($1000) [2015, 2016, 2017]
dataScientist =     [133, 132, 137]
productManager =    [127, 140, 145]
designer =          [118, 118, 127]
softwareEngineer =  [129, 131, 137]
# Salary matrix
S = np.array([dataScientist,
              productManager,
              designer,
              softwareEngineer])
# Salary increase matrix
I = np.array([[1.1, 1.2, 1.3],
              [1.0, 1.0, 1.0],
              [0.9, 0.8, 0.7],
              [1.1, 1.1, 1.1]])
# Updated salary
S2 = S * I # element wise multiplication
print(S2)
print(S2[2][0] > S[2][0])