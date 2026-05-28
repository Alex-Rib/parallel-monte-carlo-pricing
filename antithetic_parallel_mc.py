"""Pricing of a European call under Black-Scholes with piecewise volatility.

Compares an analytical reference price to three Monte Carlo estimators ->
sequential, parallel, and parallel with antithetic variates -> A
multi-core scalability benchmark is included.
"""

from __future__ import annotations

import platform
import time
import multiprocessing as mp
from dataclasses import dataclass, field


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Dataclasses for parameters and results


@dataclass(frozen=True)
class MarketParams:
    """Market and contract parameters

    Parameters ->

    S0 : float
        Spot price of the underlying at t=0
    K : float
        Strike of the option
    r : float
        Risk-free rate
    T : float
        Maturity of the option in years
    """

    S0: float = 1.0
    K: float = 1.03
    r: float = 0.02
    T: float = 1.0


@dataclass(frozen=True)
class MCConfig:
    """Numerical configuration of the Monte Carlo simulations

    Parameters ->
    n : int
        Number of time discretization steps
    N : int
        Total number of simulated trajectories
    alpha : float
        Confidence interval level (CI at 1 - alpha)
    seed : int
        Random seed for reproducibility
    """

    n: int = 128
    N: int = 10**6
    alpha: float = 0.025
    seed: int = 42  # the answer


@dataclass(frozen=True)
class MCResult:
    """Result of a Monte Carlo simulation.

    Parameters ->
    price : float
        Monte Carlo estimator of the price.
    ci_low, ci_high : float
        Lower and upper bounds of the confidence interval.
    elapsed : float
        Execution time in seconds.
    """

    price: float
    ci_low: float
    ci_high: float
    elapsed: float

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_high - self.ci_low


# Piecewise volatility and analytical pricer for validation


class PiecewiseVolatility:
    """Deterministic piecewise-linear volatility on [0, T].

    The instantaneous volatility sigma(t) takes three values depending on the
    instant t: it equals 1/10 before 1/12, follows an increasing line
    (3t/5 + 1/20) between 1/12 and 1/2, then remains constant at 7/20 beyond
    1/2.

    Since the volatility depends only on time (and not on randomness), we can
    compute the integral of sigma squared on [0, T] exactly, which allows us
    to feed the Black-Scholes closed-form formula used for validation.
    """

    T1: float = 1.0 / 12.0
    T2: float = 1.0 / 2.0

    def __call__(self, t: float) -> float:
        """Evaluate sigma(t) according to the piecewise definition."""
        if t < self.T1:
            return 1.0 / 10.0
        if t < self.T2:
            return 3.0 / 5.0 * t + 1.0 / 20.0
        return 7.0 / 20.0

    def integrated_variance(self, T: float) -> float:
        """Exact integral of sigma squared on [0, T]

        We split the integral along the three pieces of the volatility and sum
        the contribution of each piece up to the upper bound T.

        Parameters ->
        T : float
            Upper bound of integration

        Returns ->
        float
            Exact value of the integral of the instantaneous variance
        """
        if T <= 0.0:
            return 0.0

        result = 0.0

        # Segment [0, T1]: sigma^2 = 1/100
        upper = min(T, self.T1)
        result += (1.0 / 100.0) * upper
        if T <= self.T1:
            return result

        # Segment [T1, T2]: sigma^2 = (3t/5 + 1/20)^2
        # Antiderivative: 3 t^3 / 25 + 3 t^2 / 100 + t / 400
        upper = min(T, self.T2)

        def primitive(t: float) -> float:
            return 3.0 * t**3 / 25.0 + 3.0 * t**2 / 100.0 + t / 400.0

        result += primitive(upper) - primitive(self.T1)
        if T <= self.T2:
            return result

        # Segment [T2, T]: sigma^2 = (7/20)^2 = 49/400
        result += (49.0 / 400.0) * (T - self.T2)
        return result


# Black-Scholes analytical pricer for MC validation


class BlackScholesPricer:
    """Analytical pricer for a European call under Black-Scholes"""

    def __init__(self, params: MarketParams, vol: PiecewiseVolatility) -> None:
        self.params = params
        self.vol = vol

    def call_price(self) -> float:
        """Price of a European call.

        Applies the Black-Scholes closed-form formula, replacing the usual
        variance with the integral of sigma squared on [0, T], which accounts
        for the time-varying volatility.

        Returns ->
        float
        """
        p = self.params
        I_T = self.vol.integrated_variance(p.T)
        sqrt_IT = np.sqrt(I_T)
        d1 = (np.log(p.S0 / p.K) + p.r * p.T + 0.5 * I_T) / sqrt_IT
        d2 = d1 - sqrt_IT
        return p.S0 * norm.cdf(d1) - p.K * np.exp(-p.r * p.T) * norm.cdf(d2)


################################################################################
# Monte Carlo core (executed in each worker)
################################################################################


def _mc_core(args: tuple) -> tuple[float, float, int]:
    """Vectorized log-Euler simulation core, executed in a worker.

    Simulates the trajectories of a call by working on log S rather than on
    S, which makes the scheme exact here since the dynamics of log S are
    Gaussian when the volatility is deterministic. All trajectories of the
    worker are advanced simultaneously (vectorized computation) and the
    antithetic option additionally generates a mirror trajectory by flipping
    the sign of the random draw.

    Each worker receives its own PCG64 generator state obtained via jumped,
    which guarantees disjoint streams of random numbers between workers
    (https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.PCG64.jumped.html)

    We only return the aggregates necessary for the merge by the main process
    (sum of payoffs, sum of squares, effective size) in order to avoid
    the cost of transferring the raw array between processes.

    Parameters ->
    args : tuple
        (log_S0, r, T, K, N_local, antithetic, bg_state, drifts, diff_coefs).

    Returns ->
    tuple of (float, float, int)
        (sum of payoffs, sum of squared payoffs, effective size)
    """
    (
        log_S0,
        r,
        T,
        K,
        N_local,
        antithetic,
        bg_state,
        drifts,
        diff_coefs,
    ) = args

    # Restore the jumped PCG64 state for this worker
    bg = np.random.PCG64()
    bg.__setstate__(bg_state)
    rng = np.random.Generator(bg)

    discount = np.exp(-r * T)

    if antithetic:
        log_S = np.full(N_local, log_S0)
        log_S_anti = np.full(N_local, log_S0)
        for drift, diff in zip(drifts, diff_coefs):
            z = rng.standard_normal(N_local)
            log_S += drift + diff * z
            log_S_anti += drift - diff * z

        S_T = np.exp(log_S)
        S_T_anti = np.exp(log_S_anti)
        payoffs = (
            discount * 0.5 * (np.maximum(S_T - K, 0.0) + np.maximum(S_T_anti - K, 0.0))
        )
    else:
        log_S = np.full(N_local, log_S0)
        for drift, diff in zip(drifts, diff_coefs):
            log_S += drift + diff * rng.standard_normal(N_local)

        S_T = np.exp(log_S)
        payoffs = discount * np.maximum(S_T - K, 0.0)

    return float(payoffs.sum()), float((payoffs**2).sum()), N_local


# Monte Carlo engine


class MCEngine:
    """Monte Carlo engine for European option pricing.

    Simulates the evolution of the underlying on log S with a log-Euler
    scheme: at each time step we add a drift term and a diffusion term
    proportional to a Gaussian draw. The scheme is exact here because the
    dynamics of log S are Gaussian as long as the volatility remains
    deterministic.

    Parameters ->
    params : MarketParams
        Market and contract parameters
    config : MCConfig
        Numerical configuration
    vol : PiecewiseVolatility
        Deterministic volatility used for the simulation
    """

    def __init__(
        self,
        params: MarketParams,
        config: MCConfig,
        vol: PiecewiseVolatility,
    ) -> None:
        self.params = params
        self.config = config
        self.vol = vol

    # HELPERS

    def _precompute_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute drift and diffusion coefficients for the time grid

        Since the volatility is deterministic, we can compute once and for all
        the vectors of drifts and diffusion coefficients on the time grid
        and send them to the workers, rather than re-evaluating each step in each process
        """
        p, c = self.params, self.config
        h = p.T / c.n
        sqrt_h = np.sqrt(h)
        time_grid = np.arange(c.n) * h
        sigmas = np.array([self.vol(t) for t in time_grid])
        drifts = (p.r - 0.5 * sigmas**2) * h
        diff_coefs = sigmas * sqrt_h
        return drifts, diff_coefs

    def _z_quantile(self) -> float:
        """Normal quantile at 1 - alpha/2 for the CI."""
        return float(norm.ppf(1 - self.config.alpha / 2))

    def _aggregate(
        self,
        results: list[tuple[float, float, int]],
        elapsed: float,
    ) -> MCResult:
        """Aggregate the workers' partial sums into an MCResult.

        Reconstructs the global mean and variance from the sums and sums of
        squares returned by each worker, then derives the standard error and
        the confidence interval.
        """
        total_sum = sum(r[0] for r in results)
        total_sum_sq = sum(r[1] for r in results)
        total_n = sum(r[2] for r in results)

        mean = total_sum / total_n
        var = (total_sum_sq - total_n * mean**2) / (total_n - 1)
        se = np.sqrt(max(var, 0.0) / total_n)
        z_q = self._z_quantile()

        return MCResult(
            price=mean,
            ci_low=mean - z_q * se,
            ci_high=mean + z_q * se,
            elapsed=elapsed,
        )

    def _build_tasks(
        self,
        antithetic: bool,
        num_cores: int,
        N_eff: int,
    ) -> list[tuple]:
        """Split N_eff trajectories into num_cores independent tasks

        Each worker receives a PCG64 state obtained via jumped(i),
        which guarantees that the pseudo-random number streams used
        by each worker are disjoint
        """
        p = self.params
        log_S0 = np.log(p.S0)
        drifts, diff_coefs = self._precompute_grid()

        base_bg = np.random.PCG64(seed=self.config.seed)
        bg_states = [base_bg.jumped(i).__getstate__() for i in range(num_cores)]

        q, rem = divmod(N_eff, num_cores)
        counts = [q + (1 if i < rem else 0) for i in range(num_cores)]

        return [
            (
                log_S0,
                p.r,
                p.T,
                p.K,
                counts[i],
                antithetic,
                bg_states[i],
                drifts,
                diff_coefs,
            )
            for i in range(num_cores)
        ]

    # Main methods: run_sequential and run_parallel

    def run_sequential(self) -> MCResult:
        """Sequential Monte Carlo estimation (a single process).

        Simulates all trajectories of a call in a single process, time step by
        time step, then computes the discounted mean price and its confidence
        interval. Serves as a time reference for measuring speed-ups.

        Returns ->

        MCResult
            Price, confidence interval and execution time.
        """
        p, c = self.params, self.config
        h = p.T / c.n
        sqrt_h = np.sqrt(h)
        rng = np.random.default_rng(seed=c.seed)

        t0 = time.perf_counter()
        log_S = np.full(c.N, np.log(p.S0))
        for i in range(c.n):
            sig = self.vol(i * h)
            z = rng.standard_normal(c.N)
            log_S += (p.r - 0.5 * sig**2) * h + sig * sqrt_h * z

        S_T = np.exp(log_S)
        discount = np.exp(-p.r * p.T)
        payoffs = discount * np.maximum(S_T - p.K, 0.0)

        mean = float(payoffs.mean())
        # ddof = 1 for the unbiased variance (divisor N-1)
        se = float(payoffs.std(ddof=1) / np.sqrt(c.N))
        z_q = self._z_quantile()
        elapsed = time.perf_counter() - t0

        return MCResult(
            price=mean,
            ci_low=mean - z_q * se,
            ci_high=mean + z_q * se,
            elapsed=elapsed,
        )

    def run_parallel(
        self,
        num_cores: int | None = None,
        antithetic: bool = False,
    ) -> MCResult:
        """Process-parallelized Monte Carlo estimation (fork on Linux, spawn on Windows)

        Distributes the trajectories of a call over several workers, each
        simulating its batch and returning its partial sums, which the master
        process then merges. The antithetic option reduces variance by
        simulating N/2 pairs, each pair producing a trajectory and its
        mirror.

        Parameters ->

        num_cores : int, optional
            Number of workers. Defaults to mp.cpu_count()
        antithetic : bool, default False
            Enables variance reduction via antithetic variates
            In this case -> N must be even and we simulate N/2 pairs
            each pair generating two trajectories.

        Returns ->

        MCResult
            Price, confidence interval and execution time.
        """
        if num_cores is None:
            num_cores = mp.cpu_count()

        if antithetic:
            assert self.config.N % 2 == 0, "N must be even for antithetic"
            N_eff = self.config.N // 2
        else:
            N_eff = self.config.N

        tasks = self._build_tasks(antithetic, num_cores, N_eff)

        t0 = time.perf_counter()
        start_method = "fork" if platform.system() != "Windows" else "spawn"
        context = mp.get_context(start_method)
        with context.Pool(num_cores) as pool:
            results = pool.map(_mc_core, tasks)
        elapsed = time.perf_counter() - t0

        return self._aggregate(results, elapsed)


# Benchmark and plots


@dataclass
class BenchmarkResults:
    """Container for scalability benchmark results."""

    core_range: list[int] = field(default_factory=list)
    times_par: list[float] = field(default_factory=list)
    times_anti: list[float] = field(default_factory=list)
    dt_seq: float = 0.0


class Benchmark:
    """Orchestrates sequential/parallel/antithetic comparisons and plots.

    Parameters ->
    engine : MCEngine
        Configured Monte Carlo engine.
    pricer : BlackScholesPricer
        Analytical reference pricer (validation).
    """

    def __init__(self, engine: MCEngine, pricer: BlackScholesPricer) -> None:
        self.engine = engine
        self.pricer = pricer

    def compare_methods(self, n_cores: int) -> tuple[MCResult, MCResult, MCResult]:
        """Run the three variants and display a summary as a DataFrame.

        Executes the sequential, the parallel and the antithetic parallel
        with the same configuration, gathers price, confidence interval, width
        and time of each in a DataFrame, displays it with the exact reference
        price and the speed-ups relative to sequential.

        Returns ->
        tuple of MCResult
            (sequential, parallel, antithetic parallel).
        """
        exact = self.pricer.call_price()
        c = self.engine.config

        seq = self.engine.run_sequential()
        par = self.engine.run_parallel(num_cores=n_cores)
        anti = self.engine.run_parallel(num_cores=n_cores, antithetic=True)

        labels = [
            "Sequential MC",
            f"Parallel MC ({n_cores} cores)",
            "Parallel MC + antithetic",
        ]
        df = pd.DataFrame(
            {
                "price": [r.price for r in (seq, par, anti)],
                "ci_low": [r.ci_low for r in (seq, par, anti)],
                "ci_high": [r.ci_high for r in (seq, par, anti)],
                "ci_width": [r.ci_width for r in (seq, par, anti)],
                "time": [r.elapsed for r in (seq, par, anti)],
            },
            index=labels,
        )

        print(f"Exact price (Black-Scholes): {exact:.6f}")
        print(f"Number of trajectories     : {c.N:,}")
        print(f"Number of cores            : {n_cores}")
        print(df.to_string(index=True))

        if par.elapsed > 0:
            print(
                f"speed up (parallel vs sequential)   : x{seq.elapsed / par.elapsed:.1f}"
            )
        if anti.elapsed > 0:
            print(
                f"speed up (antithetic vs sequential) : x{seq.elapsed / anti.elapsed:.1f}"
            )

        return seq, par, anti

    def scalability_study(self, dt_seq: float, max_cores: int) -> BenchmarkResults:
        """Measure parallel and antithetic time for 1..max_cores cores.

        Reruns the parallel computation (and its antithetic variant) for a
        growing number of cores in order to observe how the execution time
        scales with parallelism.

        Parameters ->
        dt_seq : float
            Sequential reference time used to compute speed-ups.
        max_cores : int
            Maximum number of cores tested.

        Returns ->
        BenchmarkResults
            Lists of times and range of cores tested.
        """
        res = BenchmarkResults(dt_seq=dt_seq)
        res.core_range = list(range(1, max_cores + 1))

        for w in res.core_range:
            r_par = self.engine.run_parallel(num_cores=w)
            res.times_par.append(r_par.elapsed)

            r_anti = self.engine.run_parallel(num_cores=w, antithetic=True)
            res.times_anti.append(r_anti.elapsed)

        df = pd.DataFrame(
            {
                "cores": res.core_range,
                "time_parallel": res.times_par,
                "time_antithetic": res.times_anti,
                "speedup_parallel": [dt_seq / t for t in res.times_par],
                "speedup_antithetic": [dt_seq / t for t in res.times_anti],
            }
        )

        print(
            "\nSpeed-up benchmark by number of cores relative to the sequential version (time in seconds):"
        )
        print(df.to_string(index=False))

        return res

    def plot(self, results: BenchmarkResults) -> None:
        """Display speed-up and execution time as a function of the number of cores."""
        speedup_par = [results.dt_seq / t for t in results.times_par]
        speedup_anti = [results.dt_seq / t for t in results.times_anti]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(results.core_range, speedup_par, "o-", label="Parallel")
        axes[0].plot(
            results.core_range, speedup_anti, "s-", label="Parallel + antithetic"
        )
        axes[0].set_xlabel("Number of cores")
        axes[0].set_ylabel("Speed-up (vs sequential)")
        axes[0].set_title("Speed-up")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(results.core_range, results.times_par, "o-", label="Parallel")
        axes[1].plot(
            results.core_range,
            results.times_anti,
            "s-",
            label="Parallel + antithetic",
        )
        axes[1].axhline(
            results.dt_seq,
            color="red",
            linestyle=":",
            label=f"Sequential ({results.dt_seq:.3f}s)",
        )
        axes[1].set_xlabel("Number of cores")
        axes[1].set_ylabel("Time in seconds")
        axes[1].set_title("Execution time")
        axes[1].legend()
        axes[1].grid(True)

        c = self.engine.config
        fig.suptitle(f"Monte Carlo parallelization (N={c.N:,}, n={c.n})")
        plt.tight_layout()
        plt.show()


# Main


def main() -> None:
    params = MarketParams(S0=1.0, K=1.03, r=0.02, T=1.0)
    config = MCConfig(n=128, N=10**6, alpha=0.025, seed=42)
    vol = PiecewiseVolatility()

    pricer = BlackScholesPricer(params, vol)
    engine = MCEngine(params, config, vol)
    bench = Benchmark(engine, pricer)

    n_cores = mp.cpu_count()

    seq, _, _ = bench.compare_methods(n_cores=n_cores)
    results = bench.scalability_study(dt_seq=seq.elapsed, max_cores=n_cores)
    bench.plot(results)


if __name__ == "__main__":
    main()
