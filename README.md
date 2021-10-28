# Modified Cutting angle method

The modified Cutting angle method was developed in the context of my bachelor thesis at the Karlsruhe Institute of Technology (KIT) and is intended for those interested in global optimization algorithms in machine learning (especially supervised learning) and their application to nonlinear and nonconvex Least-squares problems. The paper is unfortunately written in German, but all numerical references in the code are clearly identifiable in the paper. 

Here I present the modified Cutting angle method based on the findings of 'Cutting angle method - a tool for constrained global optimization' (https://www.tandfonline.com/doi/abs/10.1080/10556780410001647177) by G.Beliakov.

For further questions about the code and the work I am always available: paul-niklas@hotmail.de.

Introduction:

Splines are techniques of non-linear regression from the field of supervised learning.
These functions are continuous polynomial functions defined on an interval (the interval is defined as [min(x), max(x)] in the case of data points from R^2, if x is an n-dimensional vector with data points for x values). The spline function is defined on subintervals (intervals which are subsets of [min(x), max(x)]) as polynomials of degree at most m which are connected at the boundaries of the subintervals.  The boundaries of the subintervals, i.e. the connecting points, are called 'nodes'. 
Here the question arises how such nodes are determined, since the arbitrary choice of nodes most likely extrapolates the data set insufficiently. 

The modified cutting angle method calculates the knots (the number of knots is determined in advance) so that the distance of the spline function to the y-values of the data is minimal (in the case of data points from R^2). In other words, the modified cutting angle method globally tries to solve a nonlinear least squares problem. These problems are additionally nonconvex, which further complicates the determination of a global solution.


The main modifications to the cutting angle method by G.Beliakov:

- Modified cutting angle method is applied on a different feasible ('modified Simplex')

- Auxiliary vectors for the solution and determination of the auxiliary problems are defined differently

- Update of the bounds to the optimal objective function value 

A convergence proof of the modified cutting angle method supported by 'Global Minimization of Increasing Positively Homogeneous Functions over the Unit Simplex' by A.Bagirov, A.Rubinov is additionally presented.


Code:

The modified cutting angle method can be executed with the file run_cav.py and the implemented algorithm is located in cav.py. All further explanations about the execution of the code are included in the comments of the py files.

I have also attached a dataset on which the algorithm can be tested (Pezzack_data_1.npz).
