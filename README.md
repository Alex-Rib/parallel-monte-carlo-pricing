# Parallel Monte Carlo with Antithetic Variates and Piecewise Volatility

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Finance](https://img.shields.io/badge/Finance-Derivatives-green)
![Status](https://img.shields.io/badge/Status-Educational-orange)

## 📊 Description

Pricing of a European Call by **Monte Carlo simulation** in a Black-Scholes model with **deterministic piecewise volatility**. The project compares three implementations (sequential, parallel, parallel + antithetic variates) and measures the speed-up as a function of the number of cores.

## 🎯 Objectives

- Price a European call under non-constant volatility with validation by closed-form formula
- Implement reproducible parallelization via `multiprocessing` and independent PCG64 generators
- Reduce the variance of the estimators with the antithetic variates method
- Benchmark the speed-up as a function of the number of cores

## 📐 Mathematical Model

### Piecewise volatility

The volatility is a deterministic function of time:

$$\sigma(t) = \begin{cases} 0.1 & \text{if } t < \tfrac{1}{12} \\ 0.6\,t + 0.05 & \text{if } \tfrac{1}{12} \leq t < 0.5 \\ 0.35 & \text{if } t \geq 0.5 \end{cases}$$

### Analytical price

Under this model, the call price is obtained by the Black-Scholes formula, replacing $\sigma^2 T$ with the integrated variance:

$$I_T = \int_0^T \sigma^2(t) \, dt$$

$$C_0 = S_0 \, \Phi(d_1) - K \, e^{-rT} \, \Phi(d_2)$$

with:

$$d_1 = \frac{\ln(S_0 / K) + rT + \tfrac{1}{2} I_T}{\sqrt{I_T}}, \qquad d_2 = d_1 - \sqrt{I_T}$$

The integral $I_T$ is computed analytically on each piece of $\sigma$.

### Monte Carlo simulation

Trajectories are simulated in log-price using the Euler scheme:

$$\ln S_{t_{k+1}} = \ln S_{t_k} + \left(r - \tfrac{1}{2}\sigma(t_k)^2\right) h + \sigma(t_k) \sqrt{h} \, Z_k, \qquad Z_k \sim \mathcal{N}(0,1)$$

The estimated price is:

$$\hat{C}_0 = \frac{1}{N} \sum_{i=1}^{N} e^{-rT} \max(S_T^{(i)} - K, 0)$$

### Antithetic variates

For each draw $Z_k$, two trajectories $(Z_k, -Z_k)$ are simulated simultaneously and the payoffs are averaged. Generating $N/2$ pairs (so $N$ trajectories in total), the estimator is:

$$\hat{C}_0^{\text{anti}} = \frac{1}{N/2} \sum_{i=1}^{N/2} \frac{e^{-rT}}{2} \left[ \max(S_T^{(i)} - K, 0) + \max(\tilde{S}_T^{(i)} - K, 0) \right]$$

which reduces the variance of the estimator at no additional sampling cost.

### Confidence interval

For a level $\alpha$, the confidence interval is:

$$\left[ \hat{C}_0 - z_{1-\alpha/2} \, \frac{\hat{\sigma}}{\sqrt{N}}, \quad \hat{C}_0 + z_{1-\alpha/2} \, \frac{\hat{\sigma}}{\sqrt{N}} \right]$$

## 🔧 Parallelization

Each worker receives an independent PCG64 generator obtained via the `jumped` method, guaranteeing reproducibility and the absence of correlation between random streams. Results are aggregated by sum of payoffs and sum of squares in order to recompute global mean and variance.

## 📊 Parameters

| Parameter | Value  |
|-----------|--------|
| $S_0$     | 1.0    |
| $K$       | 1.03 (OTM) |
| $r$       | 2%     |
| $T$       | 1 year |
| $n$       | 128 time steps |
| $N$       | 1,000,000 |
| $\alpha$  | 2.5% |

## 📈 Results

The script displays:

- Exact price (Black-Scholes), prices estimated by each method, confidence intervals and execution times
- Speed-up (sequential / parallel) for each number of cores

And generates two plots:

- **Speed-up** as a function of the number of cores (parallel and antithetic)
- **Execution time** as a function of the number of cores, with the sequential time as reference

## 🚀 Usage
```bash
python antithetic_parallel_mc.py
```

## 📦 Dependencies
```bash
pip install numpy scipy matplotlib pandas
```

## 👨‍💻 Author

Alexandre R. - Université Paris Cité