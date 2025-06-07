# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:32:06 2025
@author: user
"""

# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, expon, norm, poisson, binom

# Configuration du style des graphiques
sns.set(style="whitegrid")

# -------------------------------------------------------
# Fonction pour simuler des données et tracer l’histogramme
# Comparaison possible avec la densité théorique (si applicable)
# -------------------------------------------------------
def simulate_and_plot(dist_name, samples, theoretical_func=None, params={}, xlim=None):
    data = samples  # Les échantillons simulés
    mean_emp = np.mean(data)      # Moyenne empirique
    var_emp = np.var(data)        # Variance empirique

    # Tracer l’histogramme des données simulées
    plt.figure(figsize=(8, 4))
    sns.histplot(data, kde=False, stat="density", bins=50, label="Histogramme", color="skyblue")

    # Si une densité théorique est fournie, on la trace en rouge
    if theoretical_func:
        x = np.linspace(min(data) if xlim is None else xlim[0],
                        max(data) if xlim is None else xlim[1], 1000)
        plt.plot(x, theoretical_func(x, **params), label="Densité théorique", color="red")

    # Affichage du titre et de la légende
    plt.title(f"{dist_name} : Moyenne = {mean_emp:.3f}, Variance = {var_emp:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Retourner la moyenne et la variance
    return mean_emp, var_emp

# -------------------------------------------------------
# Fonction pour étudier le Théorème Central Limite (TCL)
# -------------------------------------------------------
def simulate_tcl_and_plot(distribution_func, dist_name, params, n=30, N=10000, loc=None, scale=None, xlim=None):
    # Générer N moyennes d’échantillons de taille n
    samples = [np.mean(distribution_func(**params, size=n)) for _ in range(N)]
    samples = np.array(samples)

    # Tracer l’histogramme des moyennes
    plt.figure(figsize=(6, 4))
    sns.histplot(samples, kde=True, stat="density", bins=50, color="lightblue", label="Histogramme")

    # Tracer la densité normale théorique (selon TCL)
    if loc is not None and scale is not None:
        x = np.linspace(xlim[0], xlim[1], 1000)
        plt.plot(x, norm.pdf(x, loc=loc, scale=scale), label="Densité normale", color="red")

    # Affichage du titre et configuration
    plt.title(f"Moyenne de {n} variables - {dist_name}")
    plt.legend()
    plt.xlim(xlim)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------
# Début des simulations
# -------------------------------------------------------

# Nombre total d’échantillons pour chaque loi
N = 10000

# --------------------------
# Lois de base (simulation simple + densité)
# --------------------------

# Loi uniforme entre 0 et 1
simulate_and_plot("Loi uniforme [0,1]",
                  np.random.uniform(0, 1, N),
                  theoretical_func=uniform.pdf,
                  params={"loc": 0, "scale": 1})

# Loi exponentielle avec λ = 2 (donc scale = 1/λ = 0.5)
simulate_and_plot("Loi exponentielle λ=2",
                  np.random.exponential(1/2, N),
                  theoretical_func=expon.pdf,
                  params={"scale": 1/2})

# Loi normale centrée réduite N(0,1)
simulate_and_plot("Loi normale N(0,1)",
                  np.random.normal(0, 1, N),
                  theoretical_func=norm.pdf,
                  params={"loc": 0, "scale": 1},
                  xlim=(-5, 5))

# Loi de Poisson avec λ = 3 (discrète donc pas de densité continue)
simulate_and_plot("Loi de Poisson λ=3",
                  np.random.poisson(3, N))

# Loi binomiale B(20, 0.4)
simulate_and_plot("Loi binomiale B(20, 0.4)",
                  np.random.binomial(20, 0.4, N))

# --------------------------
# Étude du Théorème Central Limite (TCL)
# --------------------------

# Taille de chaque échantillon
n = 30

# Uniforme : moyenne de 30 valeurs uniformes
simulate_tcl_and_plot(np.random.uniform, "Uniforme [0,1]",
                      params={"low": 0, "high": 1},
                      loc=0.5, scale=np.sqrt(1/12)/np.sqrt(n), xlim=(0.3, 0.7))

# Exponentielle λ=2 : moyenne de 30 exponentielles
simulate_tcl_and_plot(np.random.exponential, "Exponentielle (λ = 2)",
                      params={"scale": 1/2},
                      loc=0.5, scale=0.5/np.sqrt(n), xlim=(0.3, 0.7))

# Poisson λ=3 : moyenne de 30 Poissons
simulate_tcl_and_plot(np.random.poisson, "Poisson (λ = 3)",
                      params={"lam": 3},
                      loc=3, scale=np.sqrt(3)/np.sqrt(n), xlim=(2.2, 3.8))

# Binomiale B(20, 0.4) : moyenne de 30 tirages binomiaux
simulate_tcl_and_plot(np.random.binomial, "Binomiale B(20, 0.4)",
                      params={"n": 20, "p": 0.4},
                      loc=8, scale=np.sqrt(4.8)/np.sqrt(n), xlim=(7.2, 8.8))
