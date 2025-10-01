#!/usr/bin/env python3
"""
Script to create diagrams for reinforcement learning mechanism and DQN visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, Arrow
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_rl_mechanism_diagram():
    """Create a comprehensive diagram of the reinforcement learning mechanism."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Reinforcement Learning Mechanism for Portfolio Optimization', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Environment Box
    env_box = FancyBboxPatch((0.5, 6.5), 3, 2.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightblue', 
                            edgecolor='navy', 
                            linewidth=2)
    ax.add_patch(env_box)
    ax.text(2, 7.8, 'Market Environment', fontsize=14, fontweight='bold', ha='center')
    ax.text(2, 7.4, '• Stock Prices', fontsize=10, ha='center')
    ax.text(2, 7.1, '• Returns', fontsize=10, ha='center')
    ax.text(2, 6.8, '• Volatility', fontsize=10, ha='center')
    ax.text(2, 6.5, '• Sentiment Data', fontsize=10, ha='center')
    
    # Agent Box
    agent_box = FancyBboxPatch((6.5, 6.5), 3, 2.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', 
                              edgecolor='darkgreen', 
                              linewidth=2)
    ax.add_patch(agent_box)
    ax.text(8, 7.8, 'Advanced DQN Agent', fontsize=14, fontweight='bold', ha='center')
    ax.text(8, 7.4, '• Multi-Head Attention', fontsize=10, ha='center')
    ax.text(8, 7.1, '• Residual Blocks', fontsize=10, ha='center')
    ax.text(8, 6.8, '• Dueling Architecture', fontsize=10, ha='center')
    ax.text(8, 6.5, '• Experience Replay', fontsize=10, ha='center')
    
    # State Box
    state_box = FancyBboxPatch((0.5, 3.5), 3, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', 
                              edgecolor='orange', 
                              linewidth=2)
    ax.add_patch(state_box)
    ax.text(2, 4.8, 'State Representation', fontsize=12, fontweight='bold', ha='center')
    ax.text(2, 4.4, '• Price History (5 days)', fontsize=10, ha='center')
    ax.text(2, 4.1, '• Technical Indicators', fontsize=10, ha='center')
    ax.text(2, 3.8, '• Market Sentiment', fontsize=10, ha='center')
    ax.text(2, 3.5, '• Risk Metrics', fontsize=10, ha='center')
    
    # Action Box
    action_box = FancyBboxPatch((6.5, 3.5), 3, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', 
                               edgecolor='darkred', 
                               linewidth=2)
    ax.add_patch(action_box)
    ax.text(8, 4.8, 'Action Space', fontsize=12, fontweight='bold', ha='center')
    ax.text(8, 4.4, '• Select Stock 1', fontsize=10, ha='center')
    ax.text(8, 4.1, '• Select Stock 2', fontsize=10, ha='center')
    ax.text(8, 3.8, '• ...', fontsize=10, ha='center')
    ax.text(8, 3.5, '• Select Stock N', fontsize=10, ha='center')
    
    # Reward Box
    reward_box = FancyBboxPatch((3.5, 0.5), 3, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightpink', 
                               edgecolor='purple', 
                               linewidth=2)
    ax.add_patch(reward_box)
    ax.text(5, 1.8, 'Multi-Factor Reward', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 1.4, 'R = α·r + β·s - γ·σ', fontsize=11, ha='center', style='italic')
    ax.text(5, 1.1, '• α·r: Return Component', fontsize=10, ha='center')
    ax.text(5, 0.8, '• β·s: Sentiment Component', fontsize=10, ha='center')
    ax.text(5, 0.5, '• γ·σ: Risk Penalty', fontsize=10, ha='center')
    
    # Arrows
    # Environment to State
    arrow1 = FancyArrowPatch((2, 6.5), (2, 5.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='blue', linewidth=2)
    ax.add_patch(arrow1)
    ax.text(2.3, 6, 'Observe', fontsize=10, color='blue', fontweight='bold')
    
    # State to Agent
    arrow2 = FancyArrowPatch((3.5, 4.5), (6.5, 4.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='green', linewidth=2)
    ax.add_patch(arrow2)
    ax.text(5, 4.8, 'State Input', fontsize=10, color='green', fontweight='bold')
    
    # Agent to Action
    arrow3 = FancyArrowPatch((8, 6.5), (8, 5.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='red', linewidth=2)
    ax.add_patch(arrow3)
    ax.text(8.3, 6, 'Action', fontsize=10, color='red', fontweight='bold')
    
    # Action to Environment
    arrow4 = FancyArrowPatch((6.5, 4.5), (3.5, 4.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='orange', linewidth=2)
    ax.add_patch(arrow4)
    ax.text(5, 4.2, 'Execute', fontsize=10, color='orange', fontweight='bold')
    
    # Environment to Reward
    arrow5 = FancyArrowPatch((2, 6.5), (4, 2.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='purple', linewidth=2)
    ax.add_patch(arrow5)
    ax.text(2.5, 4.5, 'Reward', fontsize=10, color='purple', fontweight='bold')
    
    # Reward to Agent
    arrow6 = FancyArrowPatch((5, 2.5), (7, 6.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='purple', linewidth=2)
    ax.add_patch(arrow6)
    ax.text(6.5, 4.5, 'Learn', fontsize=10, color='purple', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/rl_mechanism_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dqn_architecture_diagram():
    """Create a detailed diagram of the DQN architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Advanced DQN Architecture for Portfolio Optimization', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Input Layer
    input_box = FancyBboxPatch((0.5, 8), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', 
                              edgecolor='navy', 
                              linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 8.5, 'Input State\n(85 features)', fontsize=11, fontweight='bold', ha='center')
    
    # Input Projection
    proj_box = FancyBboxPatch((3, 8), 2, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', 
                             edgecolor='darkgreen', 
                             linewidth=2)
    ax.add_patch(proj_box)
    ax.text(4, 8.5, 'Input Projection\n(256 dim)', fontsize=11, fontweight='bold', ha='center')
    
    # Multi-Head Attention
    attn_box = FancyBboxPatch((5.5, 8), 2, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightyellow', 
                             edgecolor='orange', 
                             linewidth=2)
    ax.add_patch(attn_box)
    ax.text(6.5, 8.5, 'Multi-Head\nAttention (8 heads)', fontsize=11, fontweight='bold', ha='center')
    
    # Residual Blocks
    for i in range(6):
        y_pos = 6.5 - i * 0.8
        res_box = FancyBboxPatch((1, y_pos), 3, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightcoral', 
                                edgecolor='darkred', 
                                linewidth=1.5)
        ax.add_patch(res_box)
        ax.text(2.5, y_pos + 0.3, f'Residual Block {i+1}', fontsize=10, fontweight='bold', ha='center')
    
    # Cross-Attention
    cross_box = FancyBboxPatch((5, 6.5), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightpink', 
                              edgecolor='purple', 
                              linewidth=2)
    ax.add_patch(cross_box)
    ax.text(6, 7, 'Cross-Attention\n(4 heads)', fontsize=11, fontweight='bold', ha='center')
    
    # Weighted Pooling
    pool_box = FancyBboxPatch((7.5, 6.5), 2, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightgray', 
                             edgecolor='black', 
                             linewidth=2)
    ax.add_patch(pool_box)
    ax.text(8.5, 7, 'Weighted\nPooling', fontsize=11, fontweight='bold', ha='center')
    
    # Dueling Architecture
    # Value Stream
    value_box = FancyBboxPatch((1, 3.5), 2.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightcyan', 
                              edgecolor='teal', 
                              linewidth=2)
    ax.add_patch(value_box)
    ax.text(2.25, 4.5, 'Value Stream', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.25, 4.2, '• Linear(256→256)', fontsize=9, ha='center')
    ax.text(2.25, 3.9, '• LayerNorm + ReLU', fontsize=9, ha='center')
    ax.text(2.25, 3.6, '• Linear(256→128)', fontsize=9, ha='center')
    ax.text(2.25, 3.3, '• Linear(128→1)', fontsize=9, ha='center')
    
    # Advantage Stream
    adv_box = FancyBboxPatch((4, 3.5), 2.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightsteelblue', 
                            edgecolor='steelblue', 
                            linewidth=2)
    ax.add_patch(adv_box)
    ax.text(5.25, 4.5, 'Advantage Stream', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.25, 4.2, '• Linear(256→256)', fontsize=9, ha='center')
    ax.text(5.25, 3.9, '• LayerNorm + ReLU', fontsize=9, ha='center')
    ax.text(5.25, 3.6, '• Linear(256→128)', fontsize=9, ha='center')
    ax.text(5.25, 3.3, '• Linear(128→17)', fontsize=9, ha='center')
    
    # Output Layer
    output_box = FancyBboxPatch((7, 3.5), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightgoldenrodyellow', 
                               edgecolor='goldenrod', 
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(8.25, 4.5, 'Q-Values Output', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.25, 4.2, 'V(s) + A(s,a) -', fontsize=10, ha='center', style='italic')
    ax.text(8.25, 3.9, 'mean(A(s,a))', fontsize=10, ha='center', style='italic')
    ax.text(8.25, 3.6, 'LayerNorm +', fontsize=9, ha='center')
    ax.text(8.25, 3.3, 'Dropout', fontsize=9, ha='center')
    
    # Arrows
    # Input to Projection
    arrow1 = FancyArrowPatch((2.5, 8.5), (3, 8.5), 
                            arrowstyle='->', mutation_scale=15, 
                            color='blue', linewidth=2)
    ax.add_patch(arrow1)
    
    # Projection to Attention
    arrow2 = FancyArrowPatch((5, 8.5), (5.5, 8.5), 
                            arrowstyle='->', mutation_scale=15, 
                            color='green', linewidth=2)
    ax.add_patch(arrow2)
    
    # Attention to Residual
    arrow3 = FancyArrowPatch((6.5, 8), (2.5, 7.1), 
                            arrowstyle='->', mutation_scale=15, 
                            color='orange', linewidth=2)
    ax.add_patch(arrow3)
    
    # Residual to Cross-Attention
    arrow4 = FancyArrowPatch((4, 6.2), (5, 7), 
                            arrowstyle='->', mutation_scale=15, 
                            color='red', linewidth=2)
    ax.add_patch(arrow4)
    
    # Cross-Attention to Pooling
    arrow5 = FancyArrowPatch((7, 7), (7.5, 7), 
                            arrowstyle='->', mutation_scale=15, 
                            color='purple', linewidth=2)
    ax.add_patch(arrow5)
    
    # Pooling to Value Stream
    arrow6 = FancyArrowPatch((8.5, 6.5), (2.25, 5), 
                            arrowstyle='->', mutation_scale=15, 
                            color='teal', linewidth=2)
    ax.add_patch(arrow6)
    
    # Pooling to Advantage Stream
    arrow7 = FancyArrowPatch((8.5, 6.5), (5.25, 5), 
                            arrowstyle='->', mutation_scale=15, 
                            color='steelblue', linewidth=2)
    ax.add_patch(arrow7)
    
    # Value and Advantage to Output
    arrow8 = FancyArrowPatch((3.5, 4.25), (7, 4.25), 
                            arrowstyle='->', mutation_scale=15, 
                            color='goldenrod', linewidth=2)
    ax.add_patch(arrow8)
    
    # Add feature details
    ax.text(10, 8.5, 'Key Features:', fontsize=14, fontweight='bold', ha='left')
    ax.text(10, 8, '• 6 Residual Blocks', fontsize=11, ha='left')
    ax.text(10, 7.6, '• Gating Mechanisms', fontsize=11, ha='left')
    ax.text(10, 7.2, '• Layer Normalization', fontsize=11, ha='left')
    ax.text(10, 6.8, '• Dropout Regularization', fontsize=11, ha='left')
    ax.text(10, 6.4, '• Soft Target Updates', fontsize=11, ha='left')
    ax.text(10, 6, '• Prioritized Replay', fontsize=11, ha='left')
    ax.text(10, 5.6, '• Double DQN', fontsize=11, ha='left')
    
    plt.tight_layout()
    plt.savefig('outputs/dqn_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_flow_diagram():
    """Create a diagram showing the training flow and optimization process."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'DQN Training Flow and Optimization Process', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Hyperparameter Optimization
    hyper_box = FancyBboxPatch((0.5, 5.5), 2.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', 
                              edgecolor='navy', 
                              linewidth=2)
    ax.add_patch(hyper_box)
    ax.text(1.75, 6.5, 'Hyperparameter\nOptimization', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 6.1, '• Bayesian Optimization', fontsize=9, ha='center')
    ax.text(1.75, 5.8, '• TPE Sampler', fontsize=9, ha='center')
    ax.text(1.75, 5.5, '• 50 Trials', fontsize=9, ha='center')
    
    # Architecture Design
    arch_box = FancyBboxPatch((3.5, 5.5), 2.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', 
                             edgecolor='darkgreen', 
                             linewidth=2)
    ax.add_patch(arch_box)
    ax.text(4.75, 6.5, 'Architecture\nDesign', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.75, 6.1, '• Multi-Head Attention', fontsize=9, ha='center')
    ax.text(4.75, 5.8, '• Residual Blocks', fontsize=9, ha='center')
    ax.text(4.75, 5.5, '• Dueling Network', fontsize=9, ha='center')
    
    # Training Loop
    train_box = FancyBboxPatch((6.5, 5.5), 2.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', 
                              edgecolor='orange', 
                              linewidth=2)
    ax.add_patch(train_box)
    ax.text(7.75, 6.5, 'Training Loop', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.75, 6.1, '• Experience Replay', fontsize=9, ha='center')
    ax.text(7.75, 5.8, '• Target Network', fontsize=9, ha='center')
    ax.text(7.75, 5.5, '• AdamW Optimizer', fontsize=9, ha='center')
    
    # Environment Interaction
    env_box = FancyBboxPatch((1, 3), 2.5, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='lightcoral', 
                            edgecolor='darkred', 
                            linewidth=2)
    ax.add_patch(env_box)
    ax.text(2.25, 4, 'Environment\nInteraction', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.25, 3.6, '• State Observation', fontsize=9, ha='center')
    ax.text(2.25, 3.3, '• Action Selection', fontsize=9, ha='center')
    ax.text(2.25, 3, '• Reward Calculation', fontsize=9, ha='center')
    
    # Experience Buffer
    buffer_box = FancyBboxPatch((4, 3), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightpink', 
                               edgecolor='purple', 
                               linewidth=2)
    ax.add_patch(buffer_box)
    ax.text(5.25, 4, 'Experience\nReplay Buffer', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.25, 3.6, '• Store Transitions', fontsize=9, ha='center')
    ax.text(5.25, 3.3, '• Prioritized Sampling', fontsize=9, ha='center')
    ax.text(5.25, 3, '• Batch Training', fontsize=9, ha='center')
    
    # Model Update
    update_box = FancyBboxPatch((7, 3), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightgray', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(update_box)
    ax.text(8.25, 4, 'Model Update', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.25, 3.6, '• Gradient Clipping', fontsize=9, ha='center')
    ax.text(8.25, 3.3, '• Soft Target Update', fontsize=9, ha='center')
    ax.text(8.25, 3, '• Learning Rate Decay', fontsize=9, ha='center')
    
    # Evaluation
    eval_box = FancyBboxPatch((2.5, 0.5), 2.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightcyan', 
                             edgecolor='teal', 
                             linewidth=2)
    ax.add_patch(eval_box)
    ax.text(3.75, 1.5, 'Performance\nEvaluation', fontsize=12, fontweight='bold', ha='center')
    ax.text(3.75, 1.1, '• Sharpe Ratio', fontsize=9, ha='center')
    ax.text(3.75, 0.8, '• Max Drawdown', fontsize=9, ha='center')
    ax.text(3.75, 0.5, '• Risk-Adjusted Returns', fontsize=9, ha='center')
    
    # Backtesting
    backtest_box = FancyBboxPatch((5.5, 0.5), 2.5, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightsteelblue', 
                                 edgecolor='steelblue', 
                                 linewidth=2)
    ax.add_patch(backtest_box)
    ax.text(6.75, 1.5, 'Backtesting\nAnalysis', fontsize=12, fontweight='bold', ha='center')
    ax.text(6.75, 1.1, '• Walk-Forward', fontsize=9, ha='center')
    ax.text(6.75, 0.8, '• Out-of-Sample', fontsize=9, ha='center')
    ax.text(6.75, 0.5, '• Risk Metrics', fontsize=9, ha='center')
    
    # Arrows
    arrows = [
        ((1.75, 5.5), (4.75, 5.5), 'blue', 'Optimize'),
        ((4.75, 5.5), (7.75, 5.5), 'green', 'Design'),
        ((7.75, 5.5), (2.25, 4.5), 'orange', 'Train'),
        ((2.25, 3), (5.25, 3), 'red', 'Store'),
        ((5.25, 3), (8.25, 3), 'purple', 'Sample'),
        ((8.25, 3), (3.75, 2), 'black', 'Evaluate'),
        ((3.75, 0.5), (6.75, 0.5), 'teal', 'Backtest')
    ]
    
    for start, end, color, label in arrows:
        arrow = FancyArrowPatch(start, end, 
                               arrowstyle='->', mutation_scale=15, 
                               color=color, linewidth=2)
        ax.add_patch(arrow)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.2, label, fontsize=9, color=color, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig('outputs/training_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_reward_function_diagram():
    """Create a diagram showing the multi-factor reward function."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, 'Multi-Factor Reward Function Design', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Return Component
    return_box = FancyBboxPatch((0.5, 3.5), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', 
                               edgecolor='darkgreen', 
                               linewidth=2)
    ax.add_patch(return_box)
    ax.text(1.75, 4.5, 'Return Component', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 4.2, 'α · r_t', fontsize=14, ha='center', style='italic')
    ax.text(1.75, 3.9, 'α = 1.0', fontsize=10, ha='center')
    ax.text(1.75, 3.6, 'r_t = Daily Return', fontsize=9, ha='center')
    
    # Sentiment Component
    sent_box = FancyBboxPatch((3.5, 3.5), 2.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightblue', 
                             edgecolor='navy', 
                             linewidth=2)
    ax.add_patch(sent_box)
    ax.text(4.75, 4.5, 'Sentiment Component', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.75, 4.2, 'β · s_t', fontsize=14, ha='center', style='italic')
    ax.text(4.75, 3.9, 'β = 0.3', fontsize=10, ha='center')
    ax.text(4.75, 3.6, 's_t = VADER Score', fontsize=9, ha='center')
    
    # Risk Component
    risk_box = FancyBboxPatch((6.5, 3.5), 2.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightcoral', 
                             edgecolor='darkred', 
                             linewidth=2)
    ax.add_patch(risk_box)
    ax.text(7.75, 4.5, 'Risk Penalty', fontsize=12, fontweight='bold', ha='center')
    ax.text(7.75, 4.2, 'γ · σ_t', fontsize=14, ha='center', style='italic')
    ax.text(7.75, 3.9, 'γ = 0.2', fontsize=10, ha='center')
    ax.text(7.75, 3.6, 'σ_t = Volatility', fontsize=9, ha='center')
    
    # Combined Reward
    reward_box = FancyBboxPatch((3, 1), 4, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', 
                               edgecolor='orange', 
                               linewidth=3)
    ax.add_patch(reward_box)
    ax.text(5, 2.2, 'Combined Reward Function', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 1.8, 'R_t = α·r_t + β·s_t - γ·σ_t', fontsize=16, ha='center', style='italic')
    ax.text(5, 1.4, 'R_t = 1.0·r_t + 0.3·s_t - 0.2·σ_t', fontsize=12, ha='center')
    ax.text(5, 1, 'Maximize Return + Sentiment - Risk', fontsize=11, ha='center', style='italic')
    
    # Arrows
    arrow1 = FancyArrowPatch((1.75, 3.5), (4, 2.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='green', linewidth=2)
    ax.add_patch(arrow1)
    ax.text(2.5, 3, '+', fontsize=16, color='green', fontweight='bold', ha='center')
    
    arrow2 = FancyArrowPatch((4.75, 3.5), (5, 2.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='blue', linewidth=2)
    ax.add_patch(arrow2)
    ax.text(4.9, 3, '+', fontsize=16, color='blue', fontweight='bold', ha='center')
    
    arrow3 = FancyArrowPatch((7.75, 3.5), (6, 2.5), 
                            arrowstyle='->', mutation_scale=20, 
                            color='red', linewidth=2)
    ax.add_patch(arrow3)
    ax.text(7, 3, '-', fontsize=16, color='red', fontweight='bold', ha='center')
    
    # Add explanation
    ax.text(0.5, 0.5, 'Key Benefits:', fontsize=12, fontweight='bold', ha='left')
    ax.text(0.5, 0.2, '• Balances return maximization with risk control', fontsize=10, ha='left')
    ax.text(5.5, 0.2, '• Incorporates market sentiment for enhanced decision-making', fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig('outputs/reward_function_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all diagrams."""
    print("Creating RL mechanism diagram...")
    create_rl_mechanism_diagram()
    
    print("Creating DQN architecture diagram...")
    create_dqn_architecture_diagram()
    
    print("Creating training flow diagram...")
    create_training_flow_diagram()
    
    print("Creating reward function diagram...")
    create_reward_function_diagram()
    
    print("All diagrams created successfully!")
    print("Files saved to outputs/ directory:")
    print("- rl_mechanism_diagram.png")
    print("- dqn_architecture_diagram.png")
    print("- training_flow_diagram.png")
    print("- reward_function_diagram.png")

if __name__ == "__main__":
    main()
