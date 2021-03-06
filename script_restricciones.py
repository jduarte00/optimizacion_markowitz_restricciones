# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from cvxopt.solvers import qp
from cvxopt import matrix
from cvxopt import solvers
from cvxopt.blas import dot

solvers.options["show_progress"] = False
# ---------------------------
# CARGA DE DATOS (Especificar aquí la fuente del csv donde están los retornos de los activos)
retpor = pd.read_csv(
    "/home/dum/Desktop/data/european_market_returns.csv", index_col="Date"
)


# n = 4
# S = matrix(
#     [
#         [4e-2, 6e-3, -4e-3, 0.0],
#         [6e-3, 1e-2, 0.0, 0.0],
#         [-4e-3, 0.0, 2.5e-3, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#     ]
# )
# pbar = matrix([0.12, 0.10, 0.07, 0.03])
# #G = matrix(0.0, (n, n))
# #G[:: n + 1] = -1.0
# #h = matrix(0.0, (n, 1))
# A = matrix(1.0, (1,n))
# b = matrix(1.0)
# phis = [10 ** (5.0 * t / 100 - 1.0) for t in range(100)]
# portfolios = [qp(phi * S, -pbar, A = A, b = b)["x"] for phi in phis]
# returns = [dot(pbar, x) for x in portfolios]
# risks = [np.sqrt(dot(x, S * x)) for x in portfolios]
# risks_df = np.array(risks)
# returns_df = np.array(returns)
# print(np.min(risks_df))
# print(np.max(risks_df))
# print(np.min(returns_df))
# print(np.max(returns_df))
# answers_dataframe = pd.DataFrame(
#     data={"riesgo": risks_df, "retorno": returns_df, "phi": phis}
# )
# answers_dataframe.to_csv("./markowitz_nocorto_noapal_todo_capital.csv")
# answers_dataframe.plot.line(x="riesgo", y="retorno")


# IMPUTACIÓN DE DATOS
# -----------------

# ---------------------------
# CÁLCULO DE PARÁMETROS
E = np.cov(retpor.T)
g = np.mean(retpor, axis=0)
E_matrix = matrix(E)
g_matrix = matrix(g)
n = len(g)
N = 300
phis = [10 ** (5.0 * t / N - 1.0) for t in range(N)]
# # ---------------------------
# # RESTRICCION DE GANANCIA
# retornos_ganancia_minima = np.linspace(0, 0.08, 100)
# def pesos_ganancia_minima(ganancia):
#   return ganancia * ((np.linalg.inv(E)@g)/(g.T @ np.linalg.inv(E)@g))
# portfolios_ganancia_minima = [pesos_ganancia_minima(retorno) for retorno in retornos_ganancia_minima]
# riesgo_ganancia_minima = [np.sqrt(portfolio.T @ E @ portfolio ) for portfolio in portfolios_ganancia_minima]
# rest_ganancia_minima = pd.DataFrame(data = {"riesgo":riesgo_ganancia_minima,"retorno": retornos_ganancia_minima})
# # ---------------------------


# ## RESTRICCIÓN DE SUM(x_i) = 1 SOBRE TODO I
# A = matrix(1.0, (1, n))
# b = matrix(1.0)

# portfolios = [qp(phi * E_matrix, -g_matrix, A, b)["x"] for phi in phis]
# returns = [dot(g_matrix, x) for x in portfolios]
# risks = [np.sqrt(dot(x, E_matrix * x)) for x in portfolios]
# risks_df = np.array(risks)
# returns_df = np.array(returns)
# rest_todos_activos = pd.DataFrame(
#     data={"riesgo": risks_df, "retorno": returns_df, "phi": phis}
# )
# rest_todos_activos.to_csv("./markowitz_todo_capital.csv")
# print(np.min(risks_df))
# print(np.max(risks_df))
# print(np.min(returns_df))
# print(np.max(returns_df))
# rest_todos_activos.plot.line(x="riesgo", y="retorno")
# # --------------------

# # RESTRICCIÓN DE NO VENTAS EN CORTO

# G = matrix(0.0, (n, n))
# G[:: n + 1] = -1.0
# h = matrix(0.0, (n, 1))
# portfolios = [qp(phi * E_matrix, -g_matrix, G=G, h=h)["x"] for phi in phis]
# returns = [dot(g_matrix, x) for x in portfolios]
# risks = [np.sqrt(dot(x, E_matrix * x)) for x in portfolios]
# risks_df = np.array(risks)
# returns_df = np.array(returns)
# rest_no_corto = pd.DataFrame(
#     data={"riesgo": risks_df, "retorno": returns_df, "phi": phis}
# )
# rest_no_corto.to_csv("./markowitz_no_corto.csv")
# print(np.min(risks_df))
# print(np.max(risks_df))
# print(np.min(returns_df))
# print(np.max(returns_df))
# rest_todos_activos.plot.line(x="riesgo", y="retorno")


# RESTRICCION DE NO VENTAS EN CORTO + ASIGNACIÓN DE TODO EL CAPIAL + NO APALANCAMIENTO

# RESTRICCIÓN DE x_i >= 0 para todo i
G = matrix(0.0, (n, n))
G[:: n + 1] = -1.0
h = matrix(0.0, (n, 1))

## RESTRICCIÓN DE SUM(x_i) = 1 SOBRE TODO I
A = matrix(1.0, (1, n))
b = matrix(1.0)

portfolios = [qp(phi * E_matrix, -g_matrix, G, h, A, b)["x"] for phi in phis]
returns = [dot(g_matrix, x) for x in portfolios]
risks = [np.sqrt(dot(x, E_matrix * x)) for x in portfolios]
risks_df = np.array(risks)
returns_df = np.array(returns)
print(np.min(risks_df))
print(np.max(risks_df))
print(np.min(returns_df))
print(np.max(returns_df))
answers_dataframe = pd.DataFrame(
    data={"riesgo": risks_df, "retorno": returns_df, "phi": phis}
)
answers_dataframe.to_csv("./markowitz_nocorto_noapal_todo_capital.csv")
answers_dataframe.plot.line(x="riesgo", y="retorno")

# SIN RESTRICCIÓN ALGUNA


def Rin(G, E, g):
    return G ** 2 / (g.T @ np.linalg.inv(E) @ g).flatten()


valsG = np.linspace(
    answers_dataframe["retorno"].min(), answers_dataframe["retorno"].max(), 100
)
Risk = Rin(valsG, E, g)
answers_dataframe_sin = pd.DataFrame(data={"riesgo": Risk, "retorno": valsG})
answers_dataframe_sin.to_csv("./markowitz_sin_restricciones.csv")
answers_dataframe_sin.plot.line(x="riesgo", y="retorno")
print(Risk)
