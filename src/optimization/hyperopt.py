"""
Hyperparameter optimization for the advanced DQN using Bayesian optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import asdict
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import seaborn as sns

from ..agent.dqn import AdvancedDQNAgent, AdvancedDQNConfig
from ..agent.env import MarketEnv, MarketEnvConfig
from ..data.yahoo_fetch import fetch_history_for_tickers
from ..config import DEFAULT_CONFIG


class HyperparameterOptimizer:
    """Bayesian optimization for DQN hyperparameters."""
    
    def __init__(self, n_trials: int = 50, timeout: int = 3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.results = []
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization."""
        
        # Define hyperparameter search space
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512, 1024]),
            'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
            'num_blocks': trial.suggest_categorical('num_blocks', [3, 6, 9, 12]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000, 200000, 500000]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.3),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'tau': trial.suggest_float('tau', 0.001, 0.01),
            'epsilon_decay': trial.suggest_int('epsilon_decay', 5000, 20000),
        }
        
        try:
            # Create environment
            price_data = fetch_history_for_tickers(DEFAULT_CONFIG.tickers)
            if not price_data:
                return 0.0
                
            env_config = MarketEnvConfig(price_data=price_data, window=20)
            env = MarketEnv(env_config)
            
            # Create agent with trial parameters
            agent_config = AdvancedDQNConfig(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                **params
            )
            agent = AdvancedDQNAgent(agent_config)
            
            # Train for limited episodes for optimization
            total_reward = self._train_agent(agent, env, episodes=100)
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'total_reward': total_reward,
                'params': params
            }
            self.results.append(trial_result)
            
            return total_reward
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def _train_agent(self, agent: AdvancedDQNAgent, env: MarketEnv, episodes: int) -> float:
        """Train agent for limited episodes and return total reward."""
        total_reward = 0.0
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            step = 0
            
            while not done and step < 100:  # Limit episode length
                action = agent.act(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                agent.learn()
                
                episode_reward += reward
                state = next_state
                step += 1
            
            total_reward += episode_reward
            
            # Early stopping if performance is too poor
            if episode > 20 and total_reward / (episode + 1) < -10:
                break
        
        return total_reward / episodes  # Average reward per episode
    
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        print(f"Optimization completed!")
        print(f"Best value: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': self.study,
            'results': self.results
        }
    
    def plot_optimization_results(self, save_path: str = "outputs/hyperopt_results.png"):
        """Plot hyperparameter optimization results."""
        if not self.study:
            print("No optimization results to plot. Run optimize() first.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot optimization history
        trials = self.study.trials
        values = [t.value for t in trials if t.value is not None]
        axes[0, 0].plot(values, alpha=0.7)
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            params = list(importance.keys())
            values = list(importance.values())
            
            axes[0, 1].barh(params, values)
            axes[0, 1].set_title('Parameter Importance')
            axes[0, 1].set_xlabel('Importance')
        except:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Plot learning rate vs performance
        lr_values = []
        perf_values = []
        for trial in trials:
            if trial.value is not None and 'lr' in trial.params:
                lr_values.append(trial.params['lr'])
                perf_values.append(trial.value)
        
        if lr_values:
            axes[0, 2].scatter(lr_values, perf_values, alpha=0.6)
            axes[0, 2].set_xlabel('Learning Rate')
            axes[0, 2].set_ylabel('Performance')
            axes[0, 2].set_title('Learning Rate vs Performance')
            axes[0, 2].set_xscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot hidden dimension vs performance
        hidden_dims = []
        perf_values = []
        for trial in trials:
            if trial.value is not None and 'hidden_dim' in trial.params:
                hidden_dims.append(trial.params['hidden_dim'])
                perf_values.append(trial.value)
        
        if hidden_dims:
            axes[1, 0].scatter(hidden_dims, perf_values, alpha=0.6)
            axes[1, 0].set_xlabel('Hidden Dimension')
            axes[1, 0].set_ylabel('Performance')
            axes[1, 0].set_title('Hidden Dimension vs Performance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot batch size vs performance
        batch_sizes = []
        perf_values = []
        for trial in trials:
            if trial.value is not None and 'batch_size' in trial.params:
                batch_sizes.append(trial.params['batch_size'])
                perf_values.append(trial.value)
        
        if batch_sizes:
            axes[1, 1].scatter(batch_sizes, perf_values, alpha=0.6)
            axes[1, 1].set_xlabel('Batch Size')
            axes[1, 1].set_ylabel('Performance')
            axes[1, 1].set_title('Batch Size vs Performance')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot dropout rate vs performance
        dropout_rates = []
        perf_values = []
        for trial in trials:
            if trial.value is not None and 'dropout_rate' in trial.params:
                dropout_rates.append(trial.params['dropout_rate'])
                perf_values.append(trial.value)
        
        if dropout_rates:
            axes[1, 2].scatter(dropout_rates, perf_values, alpha=0.6)
            axes[1, 2].set_xlabel('Dropout Rate')
            axes[1, 2].set_ylabel('Performance')
            axes[1, 2].set_title('Dropout Rate vs Performance')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Optimization results plotted and saved to {save_path}")
    
    def generate_report(self, save_path: str = "outputs/hyperopt_report.html"):
        """Generate comprehensive hyperparameter optimization report."""
        if not self.study:
            print("No optimization results to report. Run optimize() first.")
            return
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hyperparameter Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e7f3ff; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Hyperparameter Optimization Report</h1>
                <p>Advanced DQN Portfolio Optimization</p>
            </div>
            
            <div class="section">
                <h2>Optimization Summary</h2>
                <div class="metric">
                    <strong>Best Performance:</strong> {self.study.best_value:.4f}<br>
                    <strong>Total Trials:</strong> {len(self.study.trials)}<br>
                    <strong>Optimization Direction:</strong> Maximize<br>
                    <strong>Sampler:</strong> TPE (Tree-structured Parzen Estimator)
                </div>
            </div>
            
            <div class="section">
                <h2>Best Parameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        for param, value in self.study.best_params.items():
            html_content += f"<tr><td>{param}</td><td>{value}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Top 10 Trials</h2>
                <table>
                    <tr><th>Trial</th><th>Value</th><th>Learning Rate</th><th>Hidden Dim</th><th>Batch Size</th></tr>
        """
        
        # Sort trials by value
        sorted_trials = sorted(self.study.trials, key=lambda x: x.value or 0, reverse=True)
        for i, trial in enumerate(sorted_trials[:10]):
            if trial.value is not None:
                lr = trial.params.get('lr', 'N/A')
                hidden_dim = trial.params.get('hidden_dim', 'N/A')
                batch_size = trial.params.get('batch_size', 'N/A')
                html_content += f"<tr><td>{trial.number}</td><td>{trial.value:.4f}</td><td>{lr}</td><td>{hidden_dim}</td><td>{batch_size}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Parameter Distributions</h2>
                <p>Analysis of parameter distributions across all trials...</p>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Hyperparameter optimization report saved to {save_path}")


def run_hyperparameter_optimization(n_trials: int = 50) -> Dict[str, Any]:
    """Run hyperparameter optimization and return results."""
    optimizer = HyperparameterOptimizer(n_trials=n_trials)
    results = optimizer.optimize()
    
    # Generate visualizations and reports
    optimizer.plot_optimization_results()
    optimizer.generate_report()
    
    return results


if __name__ == "__main__":
    results = run_hyperparameter_optimization(n_trials=30)
    print(f"Optimization completed with best value: {results['best_value']:.4f}")
