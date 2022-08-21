"""
Miniproyecto 1 - Procesos de Poisson y Variables Aleatorias Gamma
Raul Jimenez 19017
Donaldo Garcia 19683
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


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
# ejercicio 1.2
# 2. Considere que usted analizará hasta un máximo de 16 huracanes este año. Grafique PMF (probability mass function) de estos eventos

poisson_pmf(16, mu=7)
# %%
# Ejercicio 1.3
# 3. Considere que usted analizará hasta un máximo de 16 huracanes este año. Grafique CDF (cumulative distribution function) de estos eventos

poisson_cdf(16, mu=7)
# %%
