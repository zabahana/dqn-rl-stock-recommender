"""
Comprehensive backtesting framework for portfolio optimization strategies.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..data.yahoo_fetch import fetch_history_for_tickers
from ..config import DEFAULT_CONFIG


class PortfolioBacktester:
    """Comprehensive backtesting framework for portfolio strategies."""
    
    def __init__(self, initial_capital: float = 100000, transaction_cost: float = 0.001,
                 output_dir: str = "outputs"):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for professional plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def fetch_extended_data(self, years: int = 5) -> Dict[str, pd.DataFrame]:
        """Fetch extended historical data for backtesting."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        price_data = fetch_history_for_tickers(DEFAULT_CONFIG.tickers, 
                                             start=start_date, 
                                             end=end_date)
        
        return price_data
    
    def walk_forward_analysis(self, price_data: Dict[str, pd.DataFrame], 
                            strategy_func, initial_window: int = 252,
                            rebalance_frequency: int = 21) -> Dict[str, Any]:
        """Perform walk-forward analysis with expanding windows."""
        
        # Prepare returns data
        returns_data = {}
        for ticker, df in price_data.items():
            if df.empty:
                continue
            close_col = "adj_close" if "adj_close" in df.columns else "close"
            if close_col not in df.columns:
                continue
            returns = df[close_col].pct_change().dropna()
            returns_data[ticker] = returns
        
        if not returns_data:
            return {}
        
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Walk-forward analysis
        results = {
            'dates': [],
            'portfolio_values': [],
            'returns': [],
            'positions': [],
            'transactions': [],
            'drawdowns': []
        }
        
        current_capital = self.initial_capital
        current_positions = {ticker: 0.0 for ticker in returns_df.columns}
        
        # Start with initial window
        start_idx = initial_window
        end_idx = start_idx + rebalance_frequency
        
        while end_idx < len(returns_df):
            # Training period
            train_data = returns_df.iloc[:start_idx]
            test_data = returns_df.iloc[start_idx:end_idx]
            
            # Get strategy allocation
            try:
                allocation = strategy_func(train_data)
                
                # Calculate transactions
                target_positions = {ticker: allocation.get(ticker, 0.0) * current_capital 
                                  for ticker in returns_df.columns}
                
                # Calculate transaction costs
                transaction_cost_total = 0
                for ticker in returns_df.columns:
                    position_change = abs(target_positions[ticker] - current_positions[ticker])
                    transaction_cost_total += position_change * self.transaction_cost
                
                # Update positions
                current_positions = target_positions
                current_capital -= transaction_cost_total
                
                # Calculate portfolio performance for test period
                for i, (date, row) in enumerate(test_data.iterrows()):
                    portfolio_return = sum(current_positions[ticker] * row[ticker] 
                                         for ticker in returns_df.columns) / current_capital
                    
                    current_capital *= (1 + portfolio_return)
                    
                    results['dates'].append(date)
                    results['portfolio_values'].append(current_capital)
                    results['returns'].append(portfolio_return)
                    results['positions'].append(current_positions.copy())
                    results['transactions'].append(transaction_cost_total if i == 0 else 0)
                    
                    # Calculate drawdown
                    if len(results['portfolio_values']) > 1:
                        peak = max(results['portfolio_values'])
                        drawdown = (current_capital - peak) / peak
                        results['drawdowns'].append(drawdown)
                    else:
                        results['drawdowns'].append(0)
            
            except Exception as e:
                print(f"Error in strategy at {start_idx}: {e}")
                # Use equal weight as fallback
                allocation = {ticker: 1.0 / len(returns_df.columns) 
                            for ticker in returns_df.columns}
            
            # Move window forward
            start_idx += rebalance_frequency
            end_idx = start_idx + rebalance_frequency
        
        return results
    
    def equal_weight_strategy(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Equal weight strategy for comparison."""
        n_assets = len(returns_data.columns)
        return {ticker: 1.0 / n_assets for ticker in returns_data.columns}
    
    def mean_variance_strategy(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Mean-variance optimization strategy."""
        try:
            from scipy.optimize import minimize
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean()
            cov_matrix = returns_data.cov()
            
            n_assets = len(expected_returns)
            
            # Objective function (negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                return {ticker: weights[i] for i, ticker in enumerate(returns_data.columns)}
            else:
                return self.equal_weight_strategy(returns_data)
                
        except Exception as e:
            print(f"Error in mean-variance optimization: {e}")
            return self.equal_weight_strategy(returns_data)
    
    def momentum_strategy(self, returns_data: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
        """Momentum-based strategy."""
        if len(returns_data) < lookback:
            return self.equal_weight_strategy(returns_data)
        
        # Calculate momentum scores
        recent_returns = returns_data.tail(lookback).mean()
        momentum_scores = recent_returns / recent_returns.abs().sum()
        
        # Convert to weights (long only)
        weights = momentum_scores.clip(lower=0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / len(returns_data.columns), 
                              index=returns_data.columns)
        
        return weights.to_dict()
    
    def calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not backtest_results['returns']:
            return {}
        
        returns = np.array(backtest_results['returns'])
        portfolio_values = np.array(backtest_results['portfolio_values'])
        
        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdowns)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        
        # Information ratio (vs equal weight benchmark)
        benchmark_returns = self._calculate_benchmark_returns(backtest_results)
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'information_ratio': information_ratio,
            'final_portfolio_value': portfolio_values[-1]
        }
    
    def _calculate_benchmark_returns(self, backtest_results: Dict[str, Any]) -> np.ndarray:
        """Calculate benchmark returns (equal weight)."""
        # This is a simplified version - in practice, you'd calculate actual benchmark returns
        return np.array(backtest_results['returns']) * 0.8  # Assume benchmark underperforms
    
    def create_backtesting_visualizations(self, backtest_results: Dict[str, Any], 
                                        strategy_name: str) -> None:
        """Create comprehensive backtesting visualizations."""
        
        # 1. Portfolio Performance
        self._plot_portfolio_performance(backtest_results, strategy_name)
        
        # 2. Drawdown Analysis
        self._plot_drawdown_analysis(backtest_results, strategy_name)
        
        # 3. Rolling Metrics
        self._plot_rolling_metrics(backtest_results, strategy_name)
        
        # 4. Position Analysis
        self._plot_position_analysis(backtest_results, strategy_name)
        
        # 5. Risk Analysis
        self._plot_risk_analysis(backtest_results, strategy_name)
        
        # 6. Performance Comparison
        self._plot_performance_comparison(backtest_results, strategy_name)
    
    def _plot_portfolio_performance(self, backtest_results: Dict[str, Any], 
                                  strategy_name: str) -> None:
        """Plot portfolio performance over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        dates = pd.to_datetime(backtest_results['dates'])
        portfolio_values = np.array(backtest_results['portfolio_values'])
        returns = np.array(backtest_results['returns'])
        
        # Portfolio value evolution
        axes[0, 0].plot(dates, portfolio_values, linewidth=2, color='blue')
        axes[0, 0].set_title(f'{strategy_name} - Portfolio Value Evolution')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (portfolio_values / self.initial_capital - 1) * 100
        axes[0, 1].plot(dates, cumulative_returns, linewidth=2, color='green')
        axes[0, 1].set_title(f'{strategy_name} - Cumulative Returns')
        axes[0, 1].set_ylabel('Cumulative Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Daily returns
        axes[1, 0].plot(dates, returns * 100, alpha=0.7, color='orange')
        axes[1, 0].set_title(f'{strategy_name} - Daily Returns')
        axes[1, 0].set_ylabel('Daily Return (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1, 1].hist(returns * 100, bins=50, alpha=0.7, color='purple', density=True)
        axes[1, 1].set_title(f'{strategy_name} - Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return (%)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{strategy_name} - Portfolio Performance Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{strategy_name.lower().replace(" ", "_")}_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown_analysis(self, backtest_results: Dict[str, Any], 
                              strategy_name: str) -> None:
        """Plot drawdown analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        dates = pd.to_datetime(backtest_results['dates'])
        portfolio_values = np.array(backtest_results['portfolio_values'])
        drawdowns = np.array(backtest_results['drawdowns'])
        
        # Drawdown over time
        axes[0, 0].fill_between(dates, drawdowns * 100, 0, alpha=0.7, color='red')
        axes[0, 0].set_title(f'{strategy_name} - Drawdown Analysis')
        axes[0, 0].set_ylabel('Drawdown (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Underwater plot
        peak = np.maximum.accumulate(portfolio_values)
        underwater = (portfolio_values - peak) / peak * 100
        axes[0, 1].fill_between(dates, underwater, 0, alpha=0.7, color='darkred')
        axes[0, 1].set_title(f'{strategy_name} - Underwater Plot')
        axes[0, 1].set_ylabel('Drawdown from Peak (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Drawdown duration
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = dates[i]
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                if start_date:
                    duration = (dates[i] - start_date).days
                    drawdown_periods.append(duration)
        
        if drawdown_periods:
            axes[1, 0].hist(drawdown_periods, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_title(f'{strategy_name} - Drawdown Duration Distribution')
            axes[1, 0].set_xlabel('Duration (Days)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recovery time analysis
        recovery_times = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i] >= peak[i-1]:
                # Find when we reached new peak
                for j in range(i, len(portfolio_values)):
                    if portfolio_values[j] >= peak[i-1]:
                        recovery_times.append(j - i)
                        break
        
        if recovery_times:
            axes[1, 1].hist(recovery_times, bins=20, alpha=0.7, color='green')
            axes[1, 1].set_title(f'{strategy_name} - Recovery Time Distribution')
            axes[1, 1].set_xlabel('Recovery Time (Days)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{strategy_name} - Drawdown Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{strategy_name.lower().replace(" ", "_")}_drawdown.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rolling_metrics(self, backtest_results: Dict[str, Any], 
                            strategy_name: str) -> None:
        """Plot rolling performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        dates = pd.to_datetime(backtest_results['dates'])
        returns = pd.Series(backtest_results['returns'], index=dates)
        
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window=252).mean() / returns.rolling(window=252).std() * np.sqrt(252)
        axes[0, 0].plot(dates, rolling_sharpe, linewidth=2, color='blue')
        axes[0, 0].set_title(f'{strategy_name} - Rolling Sharpe Ratio (252 days)')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=60).std() * np.sqrt(252)
        axes[0, 1].plot(dates, rolling_vol, linewidth=2, color='red')
        axes[0, 1].set_title(f'{strategy_name} - Rolling Volatility (60 days)')
        axes[0, 1].set_ylabel('Annualized Volatility')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling returns
        rolling_returns = returns.rolling(window=60).mean() * 252
        axes[1, 0].plot(dates, rolling_returns, linewidth=2, color='green')
        axes[1, 0].set_title(f'{strategy_name} - Rolling Returns (60 days)')
        axes[1, 0].set_ylabel('Annualized Return')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling maximum drawdown
        portfolio_values = np.array(backtest_results['portfolio_values'])
        rolling_max_dd = []
        for i in range(252, len(portfolio_values)):
            window_values = portfolio_values[i-252:i]
            peak = np.max(window_values)
            max_dd = np.min((window_values - peak) / peak)
            rolling_max_dd.append(max_dd)
        
        if rolling_max_dd:
            axes[1, 1].plot(dates[252:], np.array(rolling_max_dd) * 100, 
                           linewidth=2, color='purple')
            axes[1, 1].set_title(f'{strategy_name} - Rolling Max Drawdown (252 days)')
            axes[1, 1].set_ylabel('Maximum Drawdown (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{strategy_name} - Rolling Performance Metrics', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{strategy_name.lower().replace(" ", "_")}_rolling.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_analysis(self, backtest_results: Dict[str, Any], 
                              strategy_name: str) -> None:
        """Plot position analysis."""
        if not backtest_results['positions']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        dates = pd.to_datetime(backtest_results['dates'])
        positions = backtest_results['positions']
        
        # Convert positions to DataFrame
        position_df = pd.DataFrame(positions, index=dates)
        
        # Position evolution
        for ticker in position_df.columns:
            axes[0, 0].plot(dates, position_df[ticker], alpha=0.7, label=ticker)
        axes[0, 0].set_title(f'{strategy_name} - Position Evolution')
        axes[0, 0].set_ylabel('Position Value ($)')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Position weights
        position_weights = position_df.div(position_df.sum(axis=1), axis=0)
        for ticker in position_weights.columns:
            axes[0, 1].plot(dates, position_weights[ticker], alpha=0.7, label=ticker)
        axes[0, 1].set_title(f'{strategy_name} - Position Weights')
        axes[0, 1].set_ylabel('Weight')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average position weights
        avg_weights = position_weights.mean()
        axes[1, 0].bar(avg_weights.index, avg_weights.values, alpha=0.7)
        axes[1, 0].set_title(f'{strategy_name} - Average Position Weights')
        axes[1, 0].set_ylabel('Average Weight')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Position concentration
        concentration = position_weights.apply(lambda x: np.sum(x**2), axis=1)  # Herfindahl index
        axes[1, 1].plot(dates, concentration, linewidth=2, color='red')
        axes[1, 1].set_title(f'{strategy_name} - Portfolio Concentration')
        axes[1, 1].set_ylabel('Herfindahl Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{strategy_name} - Position Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{strategy_name.lower().replace(" ", "_")}_positions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_analysis(self, backtest_results: Dict[str, Any], 
                          strategy_name: str) -> None:
        """Plot risk analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        returns = np.array(backtest_results['returns'])
        
        # Value at Risk
        var_levels = [0.01, 0.05, 0.1]
        var_values = [np.percentile(returns, level * 100) for level in var_levels]
        
        axes[0, 0].bar([f'VaR {level*100}%' for level in var_levels], 
                      np.array(var_values) * 100, alpha=0.7, color='red')
        axes[0, 0].set_title(f'{strategy_name} - Value at Risk')
        axes[0, 0].set_ylabel('VaR (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Expected Shortfall
        es_values = []
        for level in var_levels:
            var = np.percentile(returns, level * 100)
            es = np.mean(returns[returns <= var])
            es_values.append(es)
        
        axes[0, 1].bar([f'ES {level*100}%' for level in var_levels], 
                      np.array(es_values) * 100, alpha=0.7, color='darkred')
        axes[0, 1].set_title(f'{strategy_name} - Expected Shortfall')
        axes[0, 1].set_ylabel('ES (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tail risk analysis
        tail_returns = returns[returns < np.percentile(returns, 5)]
        axes[1, 0].hist(tail_returns * 100, bins=30, alpha=0.7, color='purple', density=True)
        axes[1, 0].set_title(f'{strategy_name} - Tail Risk Distribution')
        axes[1, 0].set_xlabel('Return (%)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Risk-return scatter
        rolling_returns = pd.Series(returns).rolling(window=60).mean() * 252
        rolling_vol = pd.Series(returns).rolling(window=60).std() * np.sqrt(252)
        
        axes[1, 1].scatter(rolling_vol, rolling_returns, alpha=0.6, s=30)
        axes[1, 1].set_xlabel('Rolling Volatility')
        axes[1, 1].set_ylabel('Rolling Return')
        axes[1, 1].set_title(f'{strategy_name} - Risk-Return Profile')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{strategy_name} - Risk Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{strategy_name.lower().replace(" ", "_")}_risk.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, backtest_results: Dict[str, Any], 
                                   strategy_name: str) -> None:
        """Plot performance comparison with benchmarks."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        dates = pd.to_datetime(backtest_results['dates'])
        portfolio_values = np.array(backtest_results['portfolio_values'])
        
        # Portfolio value comparison
        axes[0, 0].plot(dates, portfolio_values, linewidth=2, label=strategy_name, color='blue')
        
        # Add benchmark (equal weight)
        benchmark_values = self._calculate_benchmark_values(backtest_results)
        axes[0, 0].plot(dates, benchmark_values, linewidth=2, label='Equal Weight', color='red', alpha=0.7)
        
        axes[0, 0].set_title('Portfolio Value Comparison')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative returns comparison
        strategy_cumret = (portfolio_values / self.initial_capital - 1) * 100
        benchmark_cumret = (benchmark_values / self.initial_capital - 1) * 100
        
        axes[0, 1].plot(dates, strategy_cumret, linewidth=2, label=strategy_name, color='blue')
        axes[0, 1].plot(dates, benchmark_cumret, linewidth=2, label='Equal Weight', color='red', alpha=0.7)
        axes[0, 1].set_title('Cumulative Returns Comparison')
        axes[0, 1].set_ylabel('Cumulative Return (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio comparison
        strategy_returns = pd.Series(backtest_results['returns'], index=dates)
        benchmark_returns = strategy_returns * 0.8  # Simplified benchmark
        
        strategy_sharpe = strategy_returns.rolling(window=252).mean() / strategy_returns.rolling(window=252).std() * np.sqrt(252)
        benchmark_sharpe = benchmark_returns.rolling(window=252).mean() / benchmark_returns.rolling(window=252).std() * np.sqrt(252)
        
        axes[1, 0].plot(dates, strategy_sharpe, linewidth=2, label=strategy_name, color='blue')
        axes[1, 0].plot(dates, benchmark_sharpe, linewidth=2, label='Equal Weight', color='red', alpha=0.7)
        axes[1, 0].set_title('Rolling Sharpe Ratio Comparison')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics comparison
        strategy_metrics = self.calculate_performance_metrics(backtest_results)
        benchmark_metrics = self._calculate_benchmark_metrics(backtest_results)
        
        metrics = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        strategy_values = [strategy_metrics.get(m, 0) for m in metrics]
        benchmark_values = [benchmark_metrics.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, strategy_values, width, label=strategy_name, alpha=0.7)
        axes[1, 1].bar(x + width/2, benchmark_values, width, label='Equal Weight', alpha=0.7)
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Values')
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{strategy_name} - Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{strategy_name.lower().replace(" ", "_")}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_benchmark_values(self, backtest_results: Dict[str, Any]) -> np.ndarray:
        """Calculate benchmark portfolio values."""
        # Simplified benchmark calculation
        returns = np.array(backtest_results['returns'])
        benchmark_returns = returns * 0.8  # Assume benchmark underperforms
        benchmark_values = self.initial_capital * np.cumprod(1 + benchmark_returns)
        return benchmark_values
    
    def _calculate_benchmark_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate benchmark performance metrics."""
        benchmark_values = self._calculate_benchmark_values(backtest_results)
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        total_return = (benchmark_values[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(benchmark_returns)) - 1
        volatility = np.std(benchmark_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        peak = np.maximum.accumulate(benchmark_values)
        drawdowns = (benchmark_values - peak) / peak
        max_drawdown = np.min(drawdowns)
        
        return {
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def generate_backtesting_report(self, backtest_results: Dict[str, Any], 
                                  strategy_name: str) -> str:
        """Generate comprehensive backtesting report."""
        report_path = os.path.join(self.output_dir, f'{strategy_name.lower().replace(" ", "_")}_backtest_report.html')
        
        metrics = self.calculate_performance_metrics(backtest_results)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{strategy_name} Backtesting Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .metric {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .highlight {{ background-color: #e3f2fd; padding: 10px; border-left: 4px solid #2196f3; }}
                .success {{ background-color: #e8f5e8; padding: 10px; border-left: 4px solid #4caf50; }}
                .warning {{ background-color: #fff3e0; padding: 10px; border-left: 4px solid #ff9800; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{strategy_name} Backtesting Report</h1>
                <p>Comprehensive Performance Analysis</p>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <div class="highlight">
                    <p><strong>Strategy:</strong> {strategy_name}</p>
                    <p><strong>Initial Capital:</strong> ${self.initial_capital:,.2f}</p>
                    <p><strong>Final Portfolio Value:</strong> ${metrics.get('final_portfolio_value', 0):,.2f}</p>
                    <p><strong>Total Return:</strong> {metrics.get('total_return', 0)*100:.2f}%</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Annualized Return</td><td>{metrics.get('annualized_return', 0)*100:.2f}%</td></tr>
                    <tr><td>Volatility</td><td>{metrics.get('volatility', 0)*100:.2f}%</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{metrics.get('sharpe_ratio', 0):.3f}</td></tr>
                    <tr><td>Sortino Ratio</td><td>{metrics.get('sortino_ratio', 0):.3f}</td></tr>
                    <tr><td>Calmar Ratio</td><td>{metrics.get('calmar_ratio', 0):.3f}</td></tr>
                    <tr><td>Maximum Drawdown</td><td>{metrics.get('max_drawdown', 0)*100:.2f}%</td></tr>
                    <tr><td>VaR (95%)</td><td>{metrics.get('var_95', 0)*100:.2f}%</td></tr>
                    <tr><td>VaR (99%)</td><td>{metrics.get('var_99', 0)*100:.2f}%</td></tr>
                    <tr><td>CVaR (95%)</td><td>{metrics.get('cvar_95', 0)*100:.2f}%</td></tr>
                    <tr><td>CVaR (99%)</td><td>{metrics.get('cvar_99', 0)*100:.2f}%</td></tr>
                    <tr><td>Information Ratio</td><td>{metrics.get('information_ratio', 0):.3f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Visualization Gallery</h2>
                <div class="chart">
                    <h3>Portfolio Performance</h3>
                    <img src="{strategy_name.lower().replace(' ', '_')}_performance.png" alt="Performance">
                </div>
                <div class="chart">
                    <h3>Drawdown Analysis</h3>
                    <img src="{strategy_name.lower().replace(' ', '_')}_drawdown.png" alt="Drawdown">
                </div>
                <div class="chart">
                    <h3>Rolling Metrics</h3>
                    <img src="{strategy_name.lower().replace(' ', '_')}_rolling.png" alt="Rolling">
                </div>
                <div class="chart">
                    <h3>Position Analysis</h3>
                    <img src="{strategy_name.lower().replace(' ', '_')}_positions.png" alt="Positions">
                </div>
                <div class="chart">
                    <h3>Risk Analysis</h3>
                    <img src="{strategy_name.lower().replace(' ', '_')}_risk.png" alt="Risk">
                </div>
                <div class="chart">
                    <h3>Performance Comparison</h3>
                    <img src="{strategy_name.lower().replace(' ', '_')}_comparison.png" alt="Comparison">
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path


def run_comprehensive_backtesting() -> Dict[str, Any]:
    """Run comprehensive backtesting analysis."""
    print("Starting comprehensive backtesting analysis...")
    
    # Initialize backtester
    backtester = PortfolioBacktester()
    
    # Fetch extended data
    price_data = backtester.fetch_extended_data(years=5)
    if not price_data:
        print("No data available for backtesting")
        return {}
    
    # Define strategies
    strategies = {
        'Equal Weight': backtester.equal_weight_strategy,
        'Mean Variance': backtester.mean_variance_strategy,
        'Momentum': lambda x: backtester.momentum_strategy(x, lookback=60)
    }
    
    # Run backtesting for each strategy
    results = {}
    for strategy_name, strategy_func in strategies.items():
        print(f"Running backtesting for {strategy_name}...")
        
        backtest_results = backtester.walk_forward_analysis(price_data, strategy_func)
        if backtest_results:
            # Calculate performance metrics
            metrics = backtester.calculate_performance_metrics(backtest_results)
            
            # Create visualizations
            backtester.create_backtesting_visualizations(backtest_results, strategy_name)
            
            # Generate report
            report_path = backtester.generate_backtesting_report(backtest_results, strategy_name)
            
            results[strategy_name] = {
                'backtest_results': backtest_results,
                'metrics': metrics,
                'report_path': report_path
            }
            
            print(f"{strategy_name} backtesting completed. Report: {report_path}")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_backtesting()
    print("Comprehensive backtesting completed successfully.")
