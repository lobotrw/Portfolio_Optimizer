import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pypfopt import expected_returns
from scipy.optimize import minimize


class portfolio_optimization:
    # All ratio/volatility metrics take ``freq`` = periods per year (365 for 24/7 crypto, 252 for stocks).
    def sharpe(src, freq=365, rf=0.0):
        d_r     = src.pct_change().dropna() - rf / freq
        return (d_r.mean() / d_r.std()) * np.sqrt(freq)


    def sortino(src, freq=365, rf=0.0):
        """Standard Sortino: downside deviation sqrt(mean(min(r, 0)^2)) over ALL periods."""
        d_r      = src.pct_change().dropna() - rf / freq
        downside = np.sqrt((d_r.clip(upper=0) ** 2).mean())
        return (d_r.mean() / downside) * np.sqrt(freq)

    def omega(src):
        d_r      = src.pct_change()
        stdv_neg = (d_r[d_r < 0]*-1).sum()
        stdv_pos = (d_r[d_r > 0]).sum()
        return (stdv_pos / stdv_neg) 


    def max_drawdown(src):
        return_series = src.pct_change()
        comp_ret      = (return_series+1).cumprod()
        peak          = comp_ret.expanding(min_periods=1).max()
        dd            = (comp_ret/peak)-1
        return dd.min()
    
    def rol_max_drawdown(src):
        return_series = src.pct_change()
        comp_ret      = (return_series+1).cumprod()
        peak          = comp_ret.expanding(min_periods=1).max()
        dd            = (comp_ret/peak)-1
        return dd

    def daily_vol(src):
        d_r = src.pct_change()
        return d_r.std()

    @staticmethod
    def infer_frequency(index):
        """365 if the data contains weekend dates (24/7 market such as crypto), else 252."""
        idx = pd.DatetimeIndex(index)
        return 365 if (idx.dayofweek >= 5).any() else 252

    @staticmethod
    def portfolio_value(prices, weights):
        """Buy-and-hold portfolio value starting at 1.0: each asset gets ``weight`` of the
        starting capital (prices normalised to their first value), not weight x price level."""
        w = pd.Series(weights, dtype=float).reindex(prices.columns).fillna(0.0)
        return (prices / prices.iloc[0]).mul(w, axis=1).sum(axis=1)

    @staticmethod
    def train_test_split(prices, train_frac):
        """Chronological split. The test set starts on the last training day so that the
        first test return is the first day after training (no overlap of returns)."""
        if train_frac >= 1.0:
            return prices, None
        cut = int(len(prices) * train_frac)
        return prices.iloc[:cut], prices.iloc[cut - 1:]

    @staticmethod
    def plot_ef(n_samples, ef, label, custom_weights=None, bounds=(0.0, 1.0)):
        w = portfolio_optimization.random_weights(ef.n_assets, n_samples, bounds)
        rets = w.dot(ef.expected_returns)
        stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
        sharpes = rets / stds
        plt.style.use("_classic_test_patch")
        fig, ax = plt.subplots(figsize=(15,7))
        ax.scatter(stds, rets, marker=".", c=sharpes, cmap="winter")

        # Portfolio performance calculation
        if custom_weights is not None:
            # For custom optimization methods (like Omega)
            portfolio_ret = ef.expected_returns.dot(custom_weights)
            portfolio_std = np.sqrt(custom_weights.T @ ef.cov_matrix @ custom_weights)
            ret_tangent, std_tangent = portfolio_ret, portfolio_std
        else:
            # For EfficientFrontier optimization methods
            ret_tangent, std_tangent, _ = ef.portfolio_performance()

        # Plot of portfolios
        ax.plot(std_tangent, ret_tangent, marker='*', color='r', markersize=12, label=label)

        # Output
        ax.set_title("Random Portfolios with Optimal Portfolio")
        ax.legend()

        st.subheader("The Modern Portfolio Theory (MPT)", help="The Modern Portfolio Theory (MPT) refers to an investment theory that allows investors to assemble an asset portfolio that maximizes expected return for a given level of risk. The theory assumes that investors are risk-averse; for a given level of expected return, investors will always prefer the less risky portfolio.")
        st.pyplot(fig)

    @staticmethod
    def random_weights(n_assets, n_samples, bounds=(0.0, 1.0), seed=0, max_rounds=50):
        """Dirichlet draws rejection-sampled to the user's weight bounds."""
        rng = np.random.default_rng(seed)
        lo, hi = bounds
        kept = []
        for _ in range(max_rounds):
            w = rng.dirichlet(np.ones(n_assets), n_samples * 4)
            kept.append(w[((w >= lo - 1e-12) & (w <= hi + 1e-12)).all(axis=1)])
            if sum(len(k) for k in kept) >= n_samples:
                break
        w = np.vstack(kept)[:n_samples]
        return w if len(w) else np.full((1, n_assets), 1 / n_assets)

    @staticmethod
    def omega_ratio(weights, returns, threshold=0):
        """
        Calculate the Omega ratio for a portfolio
        
        Parameters:
        weights (array): Portfolio weights
        returns (DataFrame): Asset returns
        threshold (float): Minimum acceptable return
        
        Returns:
        float: Negative Omega ratio (negative for minimization)
        """
        portfolio_returns = np.sum(returns * weights, axis=1)
        
        # Separate returns above and below threshold
        excess_returns = portfolio_returns - threshold
        positive_excess = np.maximum(excess_returns, 0)
        negative_excess = np.maximum(-excess_returns, 0)
        
        # Calculate expected values
        expected_positive = np.mean(positive_excess)
        expected_negative = np.mean(negative_excess)
        
        # Avoid division by zero
        if expected_negative == 0:
            return -1000  # Large negative number for optimization
        
        # Return negative omega ratio (since we're minimizing)
        return -(expected_positive / expected_negative)

    @staticmethod
    def optimize_omega(returns, threshold=0.0, bounds=(0.0, 1.0)):
        """
        Find the portfolio weights that maximize the Omega ratio
        
        Parameters:
        returns (DataFrame): Asset returns
        threshold (float): Minimum acceptable return
        bounds (tuple): (min_weight, max_weight) for each asset
        
        Returns:
        array: Optimal weights
        """
        n_assets = returns.shape[1]
        
        # Initial weights (equal weight)
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Constraints
        bounds = [bounds for _ in range(n_assets)]
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Optimize
        result = minimize(
            portfolio_optimization.omega_ratio,  # Now using class method directly
            init_weights,
            args=(returns, threshold),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x 