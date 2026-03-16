# Monte Carlo Parallèle avec Variables Antithétiques et Volatilité par Morceaux

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Derivatives-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

## 📊 Description

Pricing d'un Call Européen par **simulation Monte Carlo** dans un modèle de Black-Scholes à **volatilité déterministe par morceaux**. Le projet compare trois implémentations (séquentielle, parallèle, parallèle + variables antithétiques) et mesure le speed-up en fonction du nombre de cores.

## 🎯 Objectifs

- Pricer un call européen sous volatilité non constante avec validation par formule fermée
- Implémenter une parallélisation reproductible via `multiprocessing` et générateurs PCG64 indépendants
- Réduire la variance des estimateurs par la méthode des variables antithétiques
- Benchmarker le speed-up en fonction du nombre de cores

## 📐 Modèle Mathématique

### Volatilité par morceaux

La volatilité est une fonction déterministe du temps :

$$\sigma(t) = \begin{cases} 0.1 & \text{si } t < \tfrac{1}{12} \\ 0.6\,t + 0.05 & \text{si } \tfrac{1}{12} \leq t < 0.5 \\ 0.35 & \text{si } t \geq 0.5 \end{cases}$$

### Prix analytique

Sous ce modèle, le prix du call s'obtient par la formule de Black-Scholes en remplaçant $\sigma^2 T$ par la variance intégrée :

$$I_T = \int_0^T \sigma^2(t) \, dt$$

$$C_0 = S_0 \, \Phi(d_1) - K \, e^{-rT} \, \Phi(d_2)$$

avec :

$$d_1 = \frac{\ln(S_0 / K) + rT + \tfrac{1}{2} I_T}{\sqrt{I_T}}, \qquad d_2 = d_1 - \sqrt{I_T}$$

L'intégrale $I_T$ est calculée analytiquement sur chaque morceau de $\sigma$.

### Simulation Monte Carlo

Les trajectoires sont simulées en log-prix par le schéma d'Euler :

$$\ln S_{t_{k+1}} = \ln S_{t_k} + \left(r - \tfrac{1}{2}\sigma(t_k)^2\right) h + \sigma(t_k) \sqrt{h} \, Z_k, \qquad Z_k \sim \mathcal{N}(0,1)$$

Le prix estimé est :

$$\hat{C}_0 = \frac{1}{N} \sum_{i=1}^{N} e^{-rT} \max(S_T^{(i)} - K, 0)$$

### Variables antithétiques

Pour chaque tirage $Z_k$, on simule simultanément deux trajectoires $(Z_k, -Z_k)$ et on moyenne les payoffs :

$$\hat{C}_0^{\text{anti}} = \frac{1}{N} \sum_{i=1}^{N} \frac{e^{-rT}}{2} \left[ \max(S_T^{(i)} - K, 0) + \max(\tilde{S}_T^{(i)} - K, 0) \right]$$

ce qui réduit la variance de l'estimateur sans coût supplémentaire en tirages.

### Intervalle de confiance

Pour un niveau $\alpha$, l'intervalle de confiance est :

$$\left[ \hat{C}_0 - z_{1-\alpha/2} \, \frac{\hat{\sigma}}{\sqrt{N}}, \quad \hat{C}_0 + z_{1-\alpha/2} \, \frac{\hat{\sigma}}{\sqrt{N}} \right]$$

## 🔧 Parallélisation

Chaque worker reçoit un générateur PCG64 indépendant obtenu par la méthode `jumped`, garantissant la reproductibilité et l'absence de corrélation entre les flux aléatoires. Les résultats sont agrégés par somme des payoffs et somme des carrés pour recalculer moyenne et variance globales.

## 📊 Paramètres

| Paramètre | Valeur |
|-----------|--------|
| $S_0$ | 1.0 |
| $K$ | 1.03 (OTM) |
| $r$ | 2% |
| $T$ | 1 an |
| $n$ | 128 pas de temps |
| $N$ | 1 000 000 |
| $\alpha$ | 2.5% |

## 📈 Résultats

Le script affiche :

- Prix exact (Black-Scholes), prix estimés par chaque méthode, intervalles de confiance et temps d'exécution
- Speed-up (séquentiel / parallèle) pour chaque nombre de cores

Et génère deux graphiques :

- **Speed-up** en fonction du nombre de cores (parallèle et antithétique)
- **Temps d'exécution** en fonction du nombre de cores, avec le temps séquentiel en référence

## 🚀 Utilisation
```bash
python antithetic_parallel_mc.py
```

## 📦 Dépendances
```bash
pip install numpy scipy matplotlib
```

## 👨‍💻 Auteur

Alexandre R. - Université Paris Cité