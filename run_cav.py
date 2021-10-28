import cav
import numpy as np

'''
Here the data set can be read in. Pay attention, because glob_min must be adjusted, see below in the script.
'''
data = np.load(".npz")
x = data['x']
y = data['y']

#################################################################################################
########################## Modified cutting angle method is started ###############################
#################################################################################################

'''
x...x-component of the data set.
y...y-component of the data set.
order...Order, pay attention here, because this CAV works exclusively with cubic splines.
gamma...Known from the modified simplex of the BA.
runtime...Maximum runtime, which is known from the BA in the termination criterion.
glob_min...Approximation of a global minimum point (which is known with respect to the test data sets) is used to calculate IPH function with constant (c), be careful here,
because global minimum point must be adjusted depending on data set.The information about the actual global minimum point of g on the allowed set is used to calculate a
very good value for c. Unless approximated global minimum points are known, heuristic approaches can be chosen to estimate c.
If g (number of used nodes) is changed, also adjustment must be made.
must be made.
Approximation of global minimum points:

>> 1 knot (g = 1): [1.7712234]
>>
>> 2 knots (g = 2): [1.20443191 1.22467447]
>>
>> 3 knots (g = 3): [1.0829766  1.93316383 1.95340638]
>>
>> 4 knots (g = 4): [1.06273404 2.01413404 2.0343766  2.31777234]
>>
>> 5 knots (g = 5): [1.00200638 1.85219362 1.87243617 2.05461915 2.33801489]

L...Lipschitz constant, which is used to calculate IPH function with constant (c).
'''

min_fehler, min_knot = cav.beliakov_cav(x, y, order = 4, g = 2, gamma = 0.000001, runtime = 60*60*1.5, glob_min = [], L = 16000)
print('Smallest error after maximum runtime:',min_fehler)
print('Here is the best node found:', min_knot)
