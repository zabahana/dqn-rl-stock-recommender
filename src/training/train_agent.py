"""
Training script for the advanced DQN agent on financial market data.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from ..agent.dqn import AdvancedDQNAgent, AdvancedDQNConfig
from ..agent.env import MarketEnv, MarketEnvConfig
from ..data.yahoo_fetch import fetch_history_for_tickers
from ..config import DEFAULT_CONFIG


class RLPerformanceAnalyzer:
    """Analyzes and visualizes RL training performance."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_training_curves(self, stats: Dict[str, List[float]], save_path: str):
        """Plot training curves for loss, Q-values, and rewards."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0, 0].plot(stats['losses'], alpha=0.7, color='red')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-values
        axes[0, 1].plot(stats['q_values'], alpha=0.7, color='blue')
        axes[0, 1].set_title('Q-Values')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Average Q-Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode rewards
        if stats['episode_rewards']:
            axes[1, 0].plot(stats['episode_rewards'], alpha=0.7, color='green')
            axes[1, 0].set_title('Episode Rewards')
            axes[1, 0].set_xlabel('Episodes')
            axes[1, 0].set_ylabel('Cumulative Reward')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning progress (smoothed)
        if len(stats['losses']) > 100:
            window = 100
            smoothed_loss = pd.Series(stats['losses']).rolling(window=window).mean()
            axes[1, 1].plot(smoothed_loss, color='purple', linewidth=2)
            axes[1, 1].set_title(f'Smoothed Loss (window={window})')
            axes[1, 1].set_xlabel('Training Steps')
            axes[1, 1].set_ylabel('Smoothed Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_action_distribution(self, actions: List[int], save_path: str):
        """Plot action selection distribution."""
        plt.figure(figsize=(10, 6))
        action_counts = pd.Series(actions).value_counts().sort_index()
        action_counts.plot(kind='bar', color='skyblue', alpha=0.7)
        plt.title('Action Selection Distribution')
        plt.xlabel('Action (Stock Index)')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_portfolio_performance(self, returns: List[float], benchmark_returns: List[float], 
                                 save_path: str):
        """Plot portfolio performance vs benchmark."""
        plt.figure(figsize=(12, 8))
        
        # Cumulative returns
        portfolio_cumulative = np.cumprod(1 + np.array(returns))
        benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns))
        
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_cumulative, label='DQN Portfolio', linewidth=2, color='blue')
        plt.plot(benchmark_cumulative, label='Equal Weight Benchmark', linewidth=2, color='red')
        plt.title('Cumulative Returns Comparison')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        plt.subplot(2, 1, 2)
        window = 30
        portfolio_sharpe = pd.Series(returns).rolling(window=window).mean() / \
                          pd.Series(returns).rolling(window=window).std() * np.sqrt(252)
        benchmark_sharpe = pd.Series(benchmark_returns).rolling(window=window).mean() / \
                          pd.Series(benchmark_returns).rolling(window=window).std() * np.sqrt(252)
        
        plt.plot(portfolio_sharpe, label='DQN Sharpe', linewidth=2, color='blue')
        plt.plot(benchmark_sharpe, label='Benchmark Sharpe', linewidth=2, color='red')
        plt.title(f'Rolling Sharpe Ratio (window={window})')
        plt.xlabel('Time Steps')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def train_agent(episodes: int = 1000, save_plots: bool = True) -> Dict:
    """Train the advanced DQN agent and generate performance visualizations."""
    
    # Fetch market data
    print("Fetching market data...")
    price_data = fetch_history_for_tickers(DEFAULT_CONFIG.tickers)
    
    if not price_data:
        raise ValueError("No market data available")
    
    # Create environment
    env_config = MarketEnvConfig(price_data=price_data, window=20)
    env = MarketEnv(env_config)
    
    # Create agent
    agent_config = AdvancedDQNConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-4,
        batch_size=64,
        buffer_size=100000
    )
    agent = AdvancedDQNAgent(agent_config)
    
    # Training loop
    print(f"Training agent for {episodes} episodes...")
    episode_rewards = []
    all_actions = []
    portfolio_returns = []
    benchmark_returns = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_actions = []
        episode_returns = []
        
        done = False
        step = 0
        while not done and step < 200:  # Limit episode length
            action = agent.act(state, training=True)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            train_stats = agent.learn()
            
            episode_reward += reward
            episode_actions.append(action)
            episode_returns.append(reward)
            
            state = next_state
            step += 1
            
            if step % 100 == 0:
                print(f"Episode {episode}, Step {step}, Reward: {reward:.4f}")
        
        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)
        portfolio_returns.extend(episode_returns)
        
        # Calculate benchmark returns (equal weight)
        if episode_returns:
            benchmark_return = np.mean(episode_returns)  # Simplified benchmark
            benchmark_returns.extend([benchmark_return] * len(episode_returns))
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.4f}, "
                  f"Avg Q-Value = {train_stats.get('q_value', 0):.4f}")
    
    # Store episode rewards in agent stats
    agent.episode_rewards = episode_rewards
    
    # Generate visualizations
    if save_plots:
        analyzer = RLPerformanceAnalyzer()
        
        # Training curves
        stats = agent.get_training_stats()
        analyzer.plot_training_curves(stats, "outputs/rl_training_curves.png")
        
        # Action distribution
        analyzer.plot_action_distribution(all_actions, "outputs/action_distribution.png")
        
        # Portfolio performance
        analyzer.plot_portfolio_performance(portfolio_returns, benchmark_returns, 
                                          "outputs/portfolio_performance.png")
        
        print("Performance visualizations saved to outputs/")
    
    return {
        'episode_rewards': episode_rewards,
        'training_stats': agent.get_training_stats(),
        'final_performance': {
            'total_return': np.sum(portfolio_returns),
            'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252),
            'max_drawdown': calculate_max_drawdown(portfolio_returns)
        }
    }


def calculate_max_drawdown(returns: List[float]) -> float:
    """Calculate maximum drawdown from returns."""
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


if __name__ == "__main__":
    results = train_agent(episodes=500)
    print(f"Training completed. Final performance: {results['final_performance']}")
