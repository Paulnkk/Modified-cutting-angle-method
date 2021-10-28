'''
Packages used
'''
import numpy as np
import copy
import math
import scipy
import scipy.interpolate
import itertools
import timeit
import time
import scipy.special
'''
The following references are from my paper which was written in German. Of course, the mathematics remains unchanged.
'''


def beliakov_cav(x, y, order = 4, g = 4, gamma= 0.000001, runtime = 60*3, glob_min = [880.0, 890.0, 900.0, 910.0], L = 16000):
    '''
    Cutting Angle method from Algorithm 4.11 of the paper. Here it should be mentioned that this CAV works exclusively with
    the inner nodes, because especially the back transformation can be realized more easily.
    '''

    list_l_n = [] #\mathcal{K}
    l_m = [] #\mathcal{K}_{k = 1}^n
    L_k = [] #Empty list \mathcal{L}^K
    fehler_list = [] #List with the calculated errors for each iteration of the CAV
    f_list = [] #Using the list to calculate f_{best}
    min_list = [] #List containing all function values calculated by the CAV
    knots_list = [] #List in which all nodes are stored
    '''
    Be careful, because in this cell, in Function_iph(), matrixEtE() and rechteseite() the number of inner nodes must be adjusted.
    must be adjusted.
    '''
    n = g
    k = order - 1
    m = 0 #m aus (4.3), um die Basis-Vektoren zu initialisieren m = 1,\ldots,n

    '''
    Global minimum point (starting_knots) to be able to calculate c, see method "Function_iph". Must also be adjusted depending on
    data set and number of inner nodes.
    '''
    starting_knots = np.array(glob_min)
    #\min{x_r}
    a = min(x)
    #\max{x_r}
    b = max(x)
    '''
    Calculation of the basis vectors according to (4.41).
    '''
    for i in range(n+1):
        e_m = np.zeros(n+1)
        e_m.fill(gamma)
        e_m[m] = 1 - n * gamma
        e_m_retransformed = retransform_knots(a, b, e_m)
        print(e_m)
        f_e_m = Function_iph(x, y, e_m_retransformed, starting_knots, order, n, L)
        for j in range(n+1):
            if(j == m):
                l_m.append(e_m[j] / f_e_m)
            else:
                l_m.append(0)
        list_l_n.append(l_m)
        L_k.append(l_m)
        l_m = []
        f_list.append(f_e_m)
        m = m + 1
    '''
    Initialization of algorithm 4.11.
    '''
    L_k = [L_k]
    f_best = min(f_list)
    f_x_star = 10
    d_stop = 100
    timeout = time.time() + runtime #max Laufzeit
    differenz = 100
    it = 0
    '''
    Cutting Angle method terminates after max runtime "runtime". However, the termination criterion from (4.19) can also be used.
    or both. difference denotes f_{best} - \bar{d}.
    '''
    while(time.time() < timeout and differenz > 0.01): #Termination criterion: difference > 0.001
        print('----new iteration-----:', it)

        #Calculation of \bar{d} from Algorithm 4.11
        if(len(L_k) == 1):
            d_min = cal_d_1_element_new(L_k[0])
            i_min = 0
        else:
            d_min, i_min = cal_d_min_and_i_min_new(L_k)

        #Calculation of x_star from algorithm 4.11 with corresponding coordinate transformation
        x_star = eval_x_star_new(d_min, L_k[i_min])
        t_star = retransform_knots(a, b, x_star)
        print('Calculated knot vector:', t_star)
        knots_list.append(t_star)

        #Calculation of the error from (6.1) in the calculated node vector
        error = cal_delta_fehler(x, y ,t_star, order, g)
        fehler_list.append(error)
        print('Current smallest error:', min(fehler_list))
        print('This is the index to the smallest error:', fehler_list.index(min(fehler_list)))

        L_k_delete = L_k[i_min] #L_k[i_min] has been calculated with the gloabler minimum point is stored in L_k_delete
        L_k_minus_1 = L_k
        f_x_star = Function_iph(x, y, t_star, starting_knots, order, g, L)
        f_best = min(f_x_star, f_best) #f_best is updated

        #Smallest function value calculated so far is output
        min_list.append(f_x_star)
        min1 = min(min_list)
        imin = min_list.index(min(min_list))
        #print('Current min index:', imin)
        #print('Current min function value', min1)

        l_k = eval_l_k_new(f_x_star, x_star) #new l^K is calculated
        L_k = eval_L_k(L_k_minus_1, l_k, f_best) #L_k is updated
        it = it + 1

        #Termination criterion f_best - \bar{d} is updated
        d_stop, i_stop = cal_d_min_and_i_min_new(L_k)
        differenz = f_best - d_stop
        print('f_best - bar{d}', differenz)
    #Smallest error and associated node is output
    min1 = min(fehler_list)
    imin = fehler_list.index(min(fehler_list))
    min_knot = knots_list[imin]

    return min1, min_knot

##########################################################################################
############# Functions that are needed to evaluate function values ######################
##########################################################################################

'''
Calculation of the B-spline.
'''
def B(x, k, i, lambda1):
    if k == 0:
        return 1.0 if lambda1[i] <= x <= lambda1[i+1] else 0.0
    if lambda1[i+k] == lambda1[i]:
        c1 = 0.0
    else:
        c1 = (x - lambda1[i])/(lambda1[i+k] - lambda1[i]) * B(x, k-1, i, lambda1)
    if lambda1[i+k+1] == lambda1[i+1]:
        c2 = 0.0
    else:
        c2 = (lambda1[i+k+1] - x)/(lambda1[i+k+1] - lambda1[i+1]) * B(x, k-1, i+1, lambda1)
    return c1 + c2

'''
Calculate spline function as linear combination of B-splines.
'''
def bspline(x, lambda1, c, k):
    n = len(lambda1) - k - 1
    assert (n >= k+1)
    assert (len(c) >= n)
    return sum(c[i] * B(x, k, i, lambda1) for i in range(n))

'''
Calculation of delta from (3.1) of the paper.
'''
def Delta(x, y, coefficients, order, knots):
    I = len(x)
    delta = 0
    for index_r in range(len(x)):
        delta = delta + ((bspline(x[index_r],knots, coefficients, order - 1) - y[index_r])**2)
    return delta

'''
Calculation of the error from (6.1) of the paper.
'''
def Delta_fehler(x, y, coefficients, order, knots):
    I = len(x)
    delta = 0
    for index_r in range(len(x)):
        delta = delta + ((bspline(x[index_r],knots, coefficients, order - 1) - y[index_r])**2)
        #delta = delta + ((y[index_r] - bspline(x[index_r],knots, coefficients, degree - 1))**2)
    delta_sqrt = ((I - 1)**(-1) * delta)**(0.5)
    return delta_sqrt

'''
Calculation of the error according to (6.1). It should be noted that only the inner nodes are passed.
'''
def cal_delta_fehler(x, y, knots, order, g):
    k = order - 1
    minus_k = np.array([min(x), min(x), min(x), min(x)]) #je nach Datensatz anpassen
    g_plus_k = np.array([max(x), max(x), max(x), max(x)]) #je nach Datensatz anpassen

    minus_k = np.concatenate((minus_k, knots))
    minus_k_bis_g = minus_k
    minus_k_bis_g = np.concatenate((minus_k_bis_g, g_plus_k))
    minus_k_bis_g_plus_k = minus_k_bis_g
    coeff = qrzerlegung(x, y, minus_k_bis_g_plus_k, order, g)
    fehler = Delta_fehler(x, y, coeff, order, minus_k_bis_g_plus_k)
    return fehler

'''
Calculation of delta. It should be noted here that this method must be passed the inner and outer nodes.
'''
def Function(knots, x, y, coefficients, order, starting_knots, g):
    intervall_lower_bound = min(x)
    intervall_upper_bound = max(x)
    coefficients = qrzerlegung(x, y, knots, order, g)
    delta = Delta(x, y, coefficients, order, knots)
    result = delta
    return result

'''
Calculation of E^TE from (3.7) of the paper.
'''
def matrixEtE(x, y, knots, order, g):
    k = order - 1
    m = len(x)

    e = np.zeros((g+k+1, g+k+1))
    s = np.zeros(m)
    for i in range(g+k+1):
        for j in range(g+k+1):
            for r in range(m):
                s[r] = B(x[r], k, i, knots) * B(x[r], k, j, knots)
            summe = sum(s)
            e[j][i] = summe
    return e

'''
Calculation of the right side from (3.7) of the paper.
'''
def rechteseite(x, y, knots, order, g):
    k = order - 1
    m = len(x)

    rt = np.zeros(g+k+1)
    s = np.zeros(m)
    for i in range(g+k+1):
        for r in range(m):
            s[r] = B(x[r], k, i, knots) * y[r]
        summe1 = sum(s)
        rt[i] = summe1
    return rt

'''
LGS from (3.7) is solved using a QR decomposition.
'''
def qrzerlegung(x, y, knoten, order, g):
    e = matrixEtE(x, y, knoten, order, g)
    rt = rechteseite(x, y, knoten, order, g)
    Q,R = np.linalg.qr(e)
    q = np.array(Q)
    r = np.array(R)
    rt = np.array(rt)
    p = np.dot(np.transpose(q), rt)
    x_qr = np.linalg.lstsq(r,p)[0]
    return x_qr

'''
Compute the function f(p) from (4.39), where p is chosen from (4.30). I.e. the function is evaluated in "node coordinates".
lambda_k corresponds to the node vector at which the function is evaluated. starting_knots corresponds to
the global minimum point of \tilde{\delta}(p) s.t. p \in \mathcal{P}. starting_knots is needed to calculate c.
Only the inner knots are passed to this method.
'''
def Function_iph(x, y, lambda_k, starting_knots, order, g, L):
    k = order - 1
    #m = len(x)



    '''
    Outer nodes are added to the inner nodes (here the global minimal points), because the method Function() works with
    the inner and outer nodes.
    '''
    #Here, care should be taken to adjust the outer node vectors for a new data set
    minus_k = np.array([min(x), min(x), min(x), min(x)])
    g_plus_k = np.array([max(x), max(x), max(x), max(x)])

    minus_k = np.concatenate((minus_k, starting_knots))
    minus_k_bis_g = minus_k
    minus_k_bis_g = np.concatenate((minus_k_bis_g, g_plus_k))
    minus_k_bis_g_plus_k = minus_k_bis_g
    coeff = qrzerlegung(x, y, minus_k_bis_g_plus_k, order, g)
    #\min_{p \in \mathcal{P}} \tilde{\delta}(p) wird ausgewertet
    upper_bound = Function(minus_k_bis_g_plus_k, x, y, coeff, order, minus_k_bis_g_plus_k, g)
    '''
    Outer nodes are added to the inner nodes (here the nodes at which function is evaluated), since
    the Function() method works with the inner and outer nodes.
    '''
    #Here, care should be taken to adjust the outer node vectors for a new data set
    minus_k = np.array([min(x), min(x), min(x), min(x)])
    g_plus_k = np.array([max(x), max(x), max(x), max(x)])

    minus_k = np.concatenate((minus_k, lambda_k))
    minus_k_bis_g = minus_k
    minus_k_bis_g = np.concatenate((minus_k_bis_g, g_plus_k))
    minus_k_bis_g_plus_k = minus_k_bis_g
    coeff = qrzerlegung(x, y, minus_k_bis_g_plus_k, order, g)

    #Target function from (4.39) is output
    return Function(minus_k_bis_g_plus_k, x, y, coeff, order, minus_k_bis_g_plus_k, g) + 2*L - upper_bound

######################################################################################
################## Cutting Angle Methods functions ###################################
######################################################################################

'''
Elements of the modified simplex from (4.38) are transformed to node vectors. See (4.31).
'''
def retransform_knots(a, b, x_star):
    t = np.zeros(len(x_star) - 1)
    t[0] = a + x_star[0] * (b - a)
    i = 1
    while (i < len(x_star)-1):
        t[i] = t[i-1] + x_star[i] * (b - a)
        i = i + 1
    return t

'''
Steps 3. and 4. from Algorithm 4.11. It should be noted that with "check_condition_3_new()" Condition (3.) from the paper
and with "check_condition_2_new()" condition (2.) from the paper is checked.
'''
def eval_L_k(L_k_minus_1, l_k, f_best):
    L_k = []
    L_minus = []
    #Step 3. from algorithm 4.11
    for i in range(len(L_k_minus_1)):
        check = check_condition_3_new(L_k_minus_1[i], l_k)
        if(check == True):
            L_k_minus_1_copy = [list(x) for x in L_k_minus_1[i]]
            L_k.append(L_k_minus_1_copy)
        else:
            L_k_minus_1_copy_1 = [list(x) for x in L_k_minus_1[i]]
            L_minus.append(L_k_minus_1_copy_1)
    #Step 4. from algorithm 4.11
    L_minus_copy = [list(x) for x in L_minus]
    #Step 4. (a)
    for i in range(len(L_minus)):
        for j in range(len(L_minus[i])):
            L_minus[i][j] = l_k
            #Step 4. (b)
            check = check_condition_2_new(L_minus[i])
            d_L = cal_d_1_element_new_d(L_minus[i])
            #Step 4.(c)
            if(check == True and d_L < f_best):
                L_minus_copy_1 = [list(x) for x in L_minus[i]]
                L_k.append(L_minus_copy_1)
                L_minus[i] = [list(x) for x in L_minus_copy[i]]
            L_minus[i] = [list(x) for x in L_minus_copy[i]]
    #Step 4. (d)
    return L_k

'''
Requirement (3.) from Theorem 4.8 is checked.
'''
def check_condition_3_new(L_k_minus_1, l_k):
    diag_L = np.diag(L_k_minus_1)
    for i in range(len(diag_L)):
        if(diag_L[i] <= l_k[i]):
            return True
        else:
            continue
    return False

'''
Condition (2.) from Theorem 4.8 is checked.
'''
def check_condition_2_new(L_minus):
    diag_L_minus = np.diag(L_minus)
    for j in range(len(diag_L_minus)):
        for k in range(len(diag_L_minus)):
            if (diag_L_minus[j] - L_minus[k][j] <= 0.000000000001):
                if (k == j):
                    continue
                else:
                    return False
    return True

'''
Calculation of \bar{d} and additionally outputs index of \bar{L} \in \mathcal{L}^K to be able to calculate x^{\star}.
Method is for \vert \mathcal{L}^K \vert > 1.
'''
def cal_d_min_and_i_min_new(L_k):
    d_list = []
    d_min = 0
    i_min = 0
    a = []
    if (len(L_k) == 1):
        d_min = cal_d_1_element_new(L_k[0])
        i_min = 0
    else:
        for i in range(len(L_k)):
            diag_L_k_i = np.diag(L_k[i])
            for j in range(len(diag_L_k_i)):
                a.append(diag_L_k_i[j])
            d_list.append((np.sum(a))**(-1))
            a = []
        d_min = min(d_list)
        i_min = d_list.index(min(d_list))
    return d_min, i_min

'''
Calculation of \bar{d} and additionally outputs index of \bar{L} \in \mathcal{L}^K to be able to calculate x^{\star}.
Method is for \vert \mathcal{L}^K \vert = 1.
'''
def cal_d_1_element_new(L_k):
    d_min = 0
    i_min = 0
    a = []
    diag_start = np.diag(L_k)
    for i in range(len(diag_start)):
        a.append(diag_start[i])
    summe = (np.sum(a))
    d_min = summe**(-1)
    i_min = 0
    return d_min

'''
Calculation of x^{\star} from step 1.(b).
'''
def eval_x_star_new(d_min, L_k_min):
    x_star = []
    diag_L_min = np.diag(L_k_min)
    for i in range(len(diag_L_min)):
        x_star.append(d_min * diag_L_min[i])
    return x_star

'''
Calculation of l^K from step 2.(b).
'''
def eval_l_k_new(f_x_star, x_star):
    l_k = []
    for i in range(len(x_star)):
        l_k.append((x_star[i] / f_x_star))
    return l_k

'''
Calculation to check d_L < f_best. You can also use "cal_d_1_element_new()". Only for
added for reasons of clarity.
'''
def cal_d_1_element_new_d(L_k):
    d_min = 0
    i_min = 0
    a = []
    diag_start = np.diag(L_k)
    for i in range(len(diag_start)):
        a.append(diag_start[i])
    summe = (np.sum(a))
    d_min = summe**(-1)
    i_min = 0
    return d_min
