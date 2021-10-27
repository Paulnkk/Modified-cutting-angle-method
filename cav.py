'''
Verwendete Pakete
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

def beliakov_cav(x, y, order = 4, g = 4, gamma= 0.000001, runtime = 60*3, glob_min = [880.0, 890.0, 900.0, 910.0], L = 16000):
    '''
    Cutting-Angle-Verfahren aus Algorithmus 4.11 der BA. Hier sollte erwähnt werden, dass dieses CAV ausschließlich mit
    den inneren Knoten arbeitet, da insbesondere die Rücktransformation einfacher realisiert werden kann.
    '''

    list_l_n = [] #\mathcal{K}
    l_m = [] #\mathcal{K}_{k = 1}^n
    L_k = [] #Leere Liste \mathcal{L}^K
    fehler_list = [] #Liste mit den berechneten Fehlern für jede Iteration des CAV
    f_list = [] #Verwendung der List, um f_{best} zu berechnen
    min_list = [] #Liste, welche alle Funktionswerte beinhaltet, die das CAV berechnet
    knots_list = [] #Liste in der alle Knoten gespeichert werden
    '''
    Aufpassen, da in dieser Zelle, in Function_iph(), matrixEtE() und rechteseite() die Anzahl der inneren Knoten angepasst
    werden muss.
    '''
    n = g
    k = order - 1
    m = 0 #m aus (4.3), um die Basis-Vektoren zu initialisieren m = 1,\ldots,n

    '''
    Globaler Minimalpunkt (starting_knots), um c berechnen zu können, siehe Methode "Function_iph". Muss ebenfalls je nach
    Datensatz und Anzahl der inneren Knoten angepasst werden.
    '''
    starting_knots = np.array(glob_min)
    #\min{x_r}
    a = min(x)
    #\max{x_r}
    b = max(x)
    '''
    Berechnung der Basis-Vektoren nach (4.41).
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
    Initialisierung von Algorithmus 4.11.
    '''
    L_k = [L_k]
    f_best = min(f_list)
    f_x_star = 10
    d_stop = 100
    timeout = time.time() + runtime #max Laufzeit
    differenz = 100
    it = 0
    '''
    Cutting-Angle-Verfahren terminiert nach max Laufzeit "runtime". Es kann jedoch auch das Abbruchkriterium aus (4.19) ver-
    wendet werden oder beide. differenz bezeichnet f_{best} - \bar{d}.
    '''
    while(time.time() < timeout and differenz > 0.01): #Abbruchkriterium: differenz > 0.001
        print('----neue iteration-----:', it)

        #Berechnung von \bar{d} aus Algorithmus 4.11
        if(len(L_k) == 1):
            d_min = cal_d_1_element_new(L_k[0])
            i_min = 0
        else:
            d_min, i_min = cal_d_min_and_i_min_new(L_k)

        #Berechnung von x^{\star} aus Algorithmus 4.11 und Rücktransformation in Knoten-Koordinaten
        x_star = eval_x_star_new(d_min, L_k[i_min])
        t_star = retransform_knots(a, b, x_star)
        print('Berechneter Knoten:', t_star)
        knots_list.append(t_star)

        #Berechnung des Fehlers aus (6.1) in dem berechneten Knoten-Vektor
        error = cal_delta_fehler(x, y ,t_star, order, g)
        fehler_list.append(error)
        print('Aktuell kleinster Fehler:', min(fehler_list))
        print('Das ist der index zum kleinsten Fehler:', fehler_list.index(min(fehler_list)))

        L_k_delete = L_k[i_min] #L_k[i_min] mit dem gloabler Minimalpunkt berechnet worden ist, wird in L_k_delete gespeichert
        L_k_minus_1 = L_k
        f_x_star = Function_iph(x, y, t_star, starting_knots, order, g, L)
        f_best = min(f_x_star, f_best) #f_best wird aktualisiert

        #Kleinster bisher berechneter Funktionswert wird ausgegeben
        min_list.append(f_x_star)
        min1 = min(min_list)
        imin = min_list.index(min(min_list))
        #print('Aktueller min Index:', imin)
        #print('Aktueller min Fktswert', min1)

        l_k = eval_l_k_new(f_x_star, x_star) #neues l^K wird berechnet
        L_k = eval_L_k(L_k_minus_1, l_k, f_best) #L_k wird aktualisiert
        it = it + 1

        #Abbruchkriterium f_best - \bar{d} wird aktualisiert
        d_stop, i_stop = cal_d_min_and_i_min_new(L_k)
        differenz = f_best - d_stop
        print('f_best - bar{d}', differenz)
    #Kleinster Fehler und dazugehöriger Knoten wird ausgegeben
    min1 = min(fehler_list)
    imin = fehler_list.index(min(fehler_list))
    min_knot = knots_list[imin]

    return min1, min_knot

##########################################################################################
############# Funktionen die benötigt werden, um Funktionswerte auszuwerten ##############
##########################################################################################

'''
Berechnung des B-Splines.
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
Berechne Spline-Funktion als Linearkombination von B-Splines.
'''
def bspline(x, lambda1, c, k):
    n = len(lambda1) - k - 1
    assert (n >= k+1)
    assert (len(c) >= n)
    return sum(c[i] * B(x, k, i, lambda1) for i in range(n))

'''
Berechnung von Delta aus (3.1) der BA.
'''
def Delta(x, y, coefficients, order, knots):
    I = len(x)
    delta = 0
    for index_r in range(len(x)):
        delta = delta + ((bspline(x[index_r],knots, coefficients, order - 1) - y[index_r])**2)
    return delta

'''
Berechnung des Fehler aus (6.1) der BA.
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
Berechnung des Fehler nach (6.1). Es sei darauf hingewiesen, dass nur die inneren Knoten übergeben werden.
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
Berechnung von Delta. Es sei hierbei angemerkt, dass dieser
Methode die inneren und äußeren Knoten übergeben werden müssen.
'''
def Function(knots, x, y, coefficients, order, starting_knots, g):
    intervall_lower_bound = min(x)
    intervall_upper_bound = max(x)
    coefficients = qrzerlegung(x, y, knots, order, g)
    delta = Delta(x, y, coefficients, order, knots)
    result = delta
    return result

'''
Berechnung von E^TE aus (3.7) der BA.
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
Berechnung der rechten Seite aus (3.7) der BA.
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
LGS aus (3.7) wird mit Hilfe einer QR-Zerlegung gelöst.
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
Berechnung der Funktion f(p) aus (4.39), wobei p aus (4.30) gewählt wird. D.h die Funktion wird in "Knoten-Koordinaten"
ausgewertet. lambda_k entspricht dem Knoten-Vektor, an dem die Funktion ausgewertet wird. starting_knots entspricht
dem globalen Minimalpunkt von \tilde{\delta}(p) s.t. p \in \mathcal{P}. starting_knots wird benötigt, um c zu berechnen.
Dieser Methode werden ausschließlich die inneren Knoten übergeben.
'''
def Function_iph(x, y, lambda_k, starting_knots, order, g, L):
    k = order - 1
    #m = len(x)



    '''
    Äußere Knoten werden an die inneren Knoten (hier die globalen Minimalpunkte) ergänzt, da die Methode Function() mit
    den inneren und äußeren Knoten arbeitet.
    '''
    #Hier sollte darauf geachtet werden, dass bei einem neuen Datensatz die äußeren Knoten-Vektoren angepasst werden
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
    Äußere Knoten werden an die inneren Knoten (hier die Knoten an dem Funktion ausgewertet wird) ergänzt, da
    die Methode Function() mit den inneren und äußeren Knoten arbeitet.
    '''
    #Hier sollte darauf geachtet werden, dass bei einem neuen Datensatz die äußeren Knoten-Vektoren angepasst werden
    minus_k = np.array([min(x), min(x), min(x), min(x)])
    g_plus_k = np.array([max(x), max(x), max(x), max(x)])

    minus_k = np.concatenate((minus_k, lambda_k))
    minus_k_bis_g = minus_k
    minus_k_bis_g = np.concatenate((minus_k_bis_g, g_plus_k))
    minus_k_bis_g_plus_k = minus_k_bis_g
    coeff = qrzerlegung(x, y, minus_k_bis_g_plus_k, order, g)

    #Zielfunktion aus (4.39) wird ausgegeben
    return Function(minus_k_bis_g_plus_k, x, y, coeff, order, minus_k_bis_g_plus_k, g) + 2*L - upper_bound

#############################################################################
################## Methoden des Cutting-Angle-Verfahrens ####################
#############################################################################

'''
Elemente des modifizierten Simplex aus (4.38) werden zu Knoten-Vektoren transformiert. Siehe (4.31).
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
Schritt 3. und 4. aus Algorithmus 4.11. Es sei angemerkt, dass mit "check_condition_3_new()" Bedingung (3.) aus der BA
geprüft wird und mit "check_condition_2_new()" Bedingung (2.) aus der BA geprüft wird
'''
def eval_L_k(L_k_minus_1, l_k, f_best):
    L_k = []
    L_minus = []
    #Schritt 3. aus Algorithmus 4.11
    for i in range(len(L_k_minus_1)):
        check = check_condition_3_new(L_k_minus_1[i], l_k)
        if(check == True):
            L_k_minus_1_copy = [list(x) for x in L_k_minus_1[i]]
            L_k.append(L_k_minus_1_copy)
        else:
            L_k_minus_1_copy_1 = [list(x) for x in L_k_minus_1[i]]
            L_minus.append(L_k_minus_1_copy_1)
    #Schritt 4. aus Algorithmus 4.11
    L_minus_copy = [list(x) for x in L_minus]
    #Schritt 4. (a)
    for i in range(len(L_minus)):
        for j in range(len(L_minus[i])):
            L_minus[i][j] = l_k
            #Schritt 4. (b)
            check = check_condition_2_new(L_minus[i])
            d_L = cal_d_1_element_new_d(L_minus[i])
            #Schritt 4.(c)
            if(check == True and d_L < f_best):
                L_minus_copy_1 = [list(x) for x in L_minus[i]]
                L_k.append(L_minus_copy_1)
                L_minus[i] = [list(x) for x in L_minus_copy[i]]
            L_minus[i] = [list(x) for x in L_minus_copy[i]]
    #Schritt 4. (d)
    return L_k

'''
Bedingung (3.) aus Satz 4.8 wird geprüft.
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
Bedingung (2.) aus Satz 4.8 wird geprüft.
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
Berechnung von \bar{d} und gibt zusätzlich Index des \bar{L} \in \mathcal{L}^K aus, um x^{\star} berechnen zu können.
Methode ist für \vert \mathcal{L}^K \vert > 1.
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
Berechnung von \bar{d} und gibt zusätzlich Index des \bar{L} \in \mathcal{L}^K aus, um x^{\star} berechnen zu können.
Methode ist für \vert \mathcal{L}^K \vert = 1.
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
Berechnung von x^{\star} aus Schritt 1.(b).
'''
def eval_x_star_new(d_min, L_k_min):
    x_star = []
    diag_L_min = np.diag(L_k_min)
    for i in range(len(diag_L_min)):
        x_star.append(d_min * diag_L_min[i])
    return x_star

'''
Berechnung von l^K aus Schritt 2.(b).
'''
def eval_l_k_new(f_x_star, x_star):
    l_k = []
    for i in range(len(x_star)):
        l_k.append((x_star[i] / f_x_star))
    return l_k

'''
Berechung zur Überprüfung von d_L < f_best. Es kann auch "cal_d_1_element_new()" verwendet werden. Lediglich aus
Gründen der Übersichtlichkeit hinzugefügt.
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
