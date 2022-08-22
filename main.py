"""
Miniproyecto 1 - Procesos de Poisson y Variables Aleatorias Gamma
Raul Jimenez 19017
Donaldo Garcia 19683
"""

#%%
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma


# %%
def poisson_pmf(k, mu, decimals=5):
	k = np.arange(0, k)
	pmf = poisson.pmf(k, mu=mu)
	pmf = np.round(pmf, decimals)
	for val, prob in zip(k,pmf):
		print(f"k-value {val} has probability = {prob}")
	
	plt.plot(k, pmf, marker='o')
	plt.xlabel('i')
	plt.ylabel('Probability')
	plt.show()

# %%
def poisson_cdf(k, mu, decimals=5):
	k = np.arange(0, k)
	cdf = poisson.cdf(k, mu=mu)
	cdf = np.round(cdf, decimals)
	for val, prob in zip(k,cdf):
		print(f"k-value {val} (P<={val}) has probability = {prob}")
	plt.plot(k, cdf, marker='o')
	plt.xlabel('i')
	plt.ylabel('Cumulative Probability')
	plt.show()

# %%
# a = n
# b = lambda
def gamma_pmf(k, a, b, decimals=5):
	k = np.arange(0, k)
	pmf = gamma.pdf(k, a=a, scale=1/b)
	pmf = np.round(pmf, decimals)
	for val, prob in zip(k,pmf):
		print(f"k-value {val} has probability = {prob}")
	plt.plot(k, pmf, marker='o')
	plt.xlabel('t')
	plt.ylabel('Probability')
	plt.title(f'Gamma Distribution with n = {a} and labmda = {b}')
	plt.show()

# %%
def gamma_pdf_multiple_lambda(k, a, b, decimals=5):
	k = np.arange(0, k)
	pdf = []
	for i in b:
		val = np.round(gamma.pdf(k, a=a, scale=1/i), decimals)
		pdf.append(val)
	for i in range(len(pdf)):
		plt.plot(k, pdf[i], marker='o', label=f'lambda = {b[i]}')
	plt.xlabel('t')
	plt.ylabel('Probability')
	plt.legend()
	plt.title(f'Gamma Distribution with n = {a} and labmda = {b}')
	plt.show()

# %%
# ejercicio 1.2
# 2. Considere que usted analizará hasta un máximo de 16 huracanes este año. Grafique PMF (probability mass function) de estos eventos

poisson_pmf(16, mu=7)
# %%
# Ejercicio 1.3
# 3. Considere que usted analizará hasta un máximo de 16 huracanes este año. Grafique CDF (cumulative distribution function) de estos eventos

poisson_cdf(16, mu=7)
# %%
# Ejercicio 2.2
# Calcule y grafique la probabilidad para diferentes números de buses, yendo desde 0 hasta 100. ¿Cuál es la cantidad de buses más probable?
poisson_pmf(100, mu=2)

# %%
# Ejercicio 5.1
# gamma_pmf(20, a=3, b=2)
arra_gammas = [2, 1, 0.5]
n = 3
for element in arra_gammas:
	gamma_pmf(20, a=n, b=element)

gamma_pdf_multiple_lambda(20, a=3, b=arra_gammas)

# %%
