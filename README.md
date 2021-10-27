# Modified Cutting angle method
Modified Cutting angle method based on 'Cutting angle method - a tool for constrained global optimization' (https://www.tandfonline.com/doi/abs/10.1080/10556780410001647177) by G.Beliakov

Introduction:

Splines are techniques of non-linear regression from the field of supervised learning.
These functions are continuous polynomial functions defined on an interval (the interval is defined as [min(x), max(x)] in the case of data points from R^2, if x is an n-dimensional vector with data points for x values). The spline function is defined on subintervals (intervals which are subsets of [min(x), max(x)]) as polynomials of degree at most m which are connected at the boundaries of the subintervals.  The boundaries of the subintervals, i.e. the connecting points, are called 'nodes'. 
Here the question arises how such nodes are determined, since the arbitrary choice of nodes most likely extrapolates the data set insufficiently. 
The modified cutting angle method calculates the knots (the number of knots is determined in advance) so that the distance of the spline function to the y-values of the data is minimal. In other words, the modified cutting angle method globally tries to solve a nonlinear least squares problem.

The main modifications to the cutting angle method by G.Beliakov:

The modified cutting angle method is applied on a different feasible set than the cutting angle method by G.Beliakov.

The auxiliary vectors for the solution and determination of the auxiliary problems (which are solved in the cutting angle method) are defined differently than in the cutting angle method by G.Beliakov.

Code:

The modified cutting angle method can be executed with the file run_cav.py and the implemented algorithm is located in cav.py. All further explanations about the execution of the code are included in the comments of the py files.

I have also attached a dataset on which the algorithm can be tested (Pezzack_data_1.npz).


If you have any further questions about my work, please send me an email at: 
paul-niklas@hotmail.de
