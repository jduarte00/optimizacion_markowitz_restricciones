import pandas as pd
import numpy as np
from cvxopt.solvers import qp
from cvxopt import matrix
from cvxopt import solvers
from cvxopt.blas import dot

# ---------------------------
# CARGA DE DATOS
retpor = pd.read_csv('./retpor.csv', index_col=  'Date')
# ---------------------------
# CÁLCULO DE PARÁMETROS
E = np.cov(retpor.T)
g = np.mean(retpor, axis = 0)
print(np.shape(E))
print(np.shape(g))
E_matrix = matrix(E)
g_matrix = matrix(g)
print(E_matrix)
print(g_matrix)
n = len(g)
N = 100
phis = [10**(5.0*t/N-1.0) for t in range(N)]
# ---------------------------
# RESTRICCION DE GANANCIA
retornos_ganancia_minima = np.linspace(0, 0.08, 100)
def pesos_ganancia_minima(ganancia):
  return ganancia * ((np.linalg.inv(E)@g)/(g.T @ np.linalg.inv(E)@g))
portfolios_ganancia_minima = [pesos_ganancia_minima(retorno) for retorno in retornos_ganancia_minima]
riesgo_ganancia_minima = [np.sqrt(portfolio.T @ E @ portfolio ) for portfolio in portfolios_ganancia_minima]
rest_ganancia_minima = pd.DataFrame(data = {"riesgo":riesgo_ganancia_minima,"retorno": retornos_ganancia_minima})
# ---------------------------
# RESTRICCION MINIMA VARIANZA
A = matrix(1.0, (1,n))
b = matrix(1.0)
portfolios = [ qp(P = phi*E_matrix, q = -g_matrix,A= A, b =b)['x'] for phi in phis ]
returns = [ dot(g_matrix,x) for x in portfolios ]
risks = [ np.sqrt(dot(x, E_matrix*x)) for x in portfolios ]
risks_df = np.array(risks)
returns_df = np.array(returns)
rest_todo_capital = pd.DataFrame(data = {"riesgo":risks_df,"retorno": returns_df, "phi": phis})
print(rest_todo_capital.head(10))