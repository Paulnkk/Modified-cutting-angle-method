import cav
import numpy as np

'''
Hier kann der Datensatz eingelesen werden. Aufpassen, da glob_min angepasst werden muss, siehe unten im Skript.
'''
data = np.load("titanium_heat_data_1.npz")
x = data['x']
y = data['y']

#################################################################################################
########################## Cutting-Angle-Verfahren wird gestartet ###############################
#################################################################################################

'''
x...x-Komponente des Datensatzes.
y...y-Komponente des Datensatzes.
order...Ordnung, hier aufpassen, da dieses CAV ausschließlich mit kubischen Splines arbeitet.
gamma...Bekannt aus dem modifizierten Simplex der BA.
runtime...Maximale Laufzeit, welche im Abbruchkriterium aus der BA bekannt ist.
glob_min...Globaler Minimalpunkt, welcher verwendet wird, um IPH Funktion mit Konstante (c) berechnen zu können, hier aufpassen,
da globaler Minimalpunkt je nach Datensatz angepasst werden muss. Wenn g verändert wird, muss ebenfalls Anpassung vorgenommen
werden.
L...Lipschitz-Konstante, welche verwendet wird, um IPH Funktion mit Konstante (c) berechnen zu können.
'''

min_fehler, min_knot = cav.beliakov_cav(x, y, order = 4, g = 2, gamma = 0.000001, runtime = 60*60*1.5, glob_min = [880.0, 890.0], L = 16000)
print('Kleinster Fehler nach maximaler Laufzeit:',min_fehler)
print('Hier ist der beste gefundene Knoten:', min_knot)
