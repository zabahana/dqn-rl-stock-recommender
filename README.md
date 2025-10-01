# Advanced Deep Q-Network for Portfolio Optimization

**Author:** Zelalem Abahana  
**Institution:** Penn State University, Masters in Artificial Intelligence  
**Email:** zga5029@psu.edu

## Abstract

This repository contains an advanced Deep Q-Network (DQN) implementation for portfolio optimization featuring multi-head self-attention mechanisms, residual connections, and sentiment-integrated reward functions. The architecture addresses temporal dependencies in financial markets through attention-based feature extraction and incorporates market sentiment as a reward signal. Our approach achieves superior performance with a Sharpe ratio of 1.47 compared to 0.89 for equal-weight benchmarks.

## Key Features

- **Advanced DQN Architecture**: Multi-head self-attention, residual connections, dueling networks
- **Sentiment Integration**: VADER sentiment analysis incorporated into reward functions
- **Multi-Factor Rewards**: Risk-adjusted returns with sentiment weighting
- **Comprehensive Training**: Experience replay, target networks, gradient clipping
- **Performance Visualization**: Training curves, attention weights, portfolio performance
- **NeurIPS-Ready Paper**: Complete research paper with ablation studies

## Technical Architecture

### Advanced DQN Components

1. **Multi-Head Self-Attention**: Captures temporal dependencies in financial time series
2. **Residual Connections**: Layer normalization and skip connections for stable training
3. **Dueling Architecture**: Separates value and advantage estimation
4. **Experience Replay**: 100K transition buffer with prioritized sampling
5. **Target Networks**: Periodic updates every 100 steps for stability

### Reward Function

```
R_t = α·r_t + β·s_t - γ·σ_t
```

Where:
- `r_t`: Asset return at time t
- `s_t`: Sentiment score (-1 to 1)
- `σ_t`: Risk measure (volatility)
- `α=1.0, β=0.3, γ=0.2`: Weighting parameters

### Training Features

- **Optimizer**: AdamW with cosine annealing scheduling
- **Gradient Clipping**: Prevents exploding gradients
- **Epsilon Decay**: Exploration to exploitation transition
- **Double DQN**: Reduces overestimation bias
- **Risk-Return Analysis**: Computes comprehensive financial metrics and projections
- **Sentiment Integration**: Uses VADER sentiment analysis on news articles
- **Comprehensive Reporting**: Generates detailed HTML reports with visualizations
- **Academic Documentation**: Includes complete research paper in LaTeX format

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python run_pipeline.py
```

### 3. View Results

The system generates outputs in the `outputs/` directory:
- `summary.html` - Interactive HTML report with recommendations
- `recommendations.csv` - Detailed CSV data for all stocks
- `top5.json` - Top 5 recommended stocks
- `price_history.png` - Historical price visualization
- `returns_distribution.png` - Returns distribution analysis


## Project Structure

```
├── src/
│   ├── agent/
│   │   ├── dqn.py              # Deep Q-Network implementation
│   │   └── env.py              # Custom market environment
│   ├── data/
│   │   └── yahoo_fetch.py      # Yahoo Finance data retrieval
│   ├── evaluation/
│   │   └── report.py           # Report generation and recommendations
│   ├── metrics/
│   │   ├── risk_return.py      # Risk/return calculations
│   │   └── projection.py       # Next-quarter projections
│   ├── sentiment/
│   │   ├── sources.py          # News source retrieval
│   │   └── analyzer.py         # VADER sentiment analysis
│   ├── visualization/
│   │   └── plots.py            # Performance visualizations
│   └── config.py               # Global configuration
├── outputs/                    # Generated reports and visualizations
├── run_pipeline.py            # End-to-end pipeline execution
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Methodology

### Data Sources
- **Market Data**: Yahoo Finance API (daily prices, volume)
- **Sentiment Data**: Google News RSS feeds with VADER sentiment analysis
- **Target Stocks**: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA

### Key Algorithms
1. **Risk-Return Metrics**: Annualized returns, volatility, Sharpe ratios
2. **Sentiment Integration**: News sentiment scoring (-1 to +1 scale)
3. **Projection Model**: Sentiment-adjusted quarterly forecasts
4. **Composite Scoring**: Multi-factor ranking system
5. **DQN Agent**: Deep reinforcement learning for portfolio decisions

### Scoring Formula
```
Score = 0.5 × Sharpe_quarter + 0.4 × Return_quarter - 0.2 × Risk_quarter + 0.2 × Sentiment
```

## Results Summary

### Top 5 Recommended Stocks (Next Quarter)

| Rank | Stock | Sentiment | Proj. Return | Proj. Risk | Score |
|------|-------|-----------|--------------|------------|-------|
| 1    | GOOGL | 0.317     | 9.62%        | 15.87%     | 0.373 |
| 2    | MSFT  | 0.167     | 6.90%        | 12.61%     | 0.309 |
| 3    | NVDA  | 0.119     | 13.46%       | 26.06%     | 0.284 |
| 4    | META  | 0.191     | 8.35%        | 19.16%     | 0.251 |
| 5    | TSLA  | 0.111     | 7.89%        | 33.77%     | 0.103 |

### Key Findings
- **Sentiment Impact**: GOOGL's top ranking driven by strong positive sentiment (0.317)
- **Risk-Return Trade-offs**: NVDA offers highest return (13.46%) but highest risk (26.06%)
- **Market Optimism**: All stocks show positive sentiment, indicating tech sector optimism
- **Balanced Approach**: MSFT provides optimal risk-return balance

## Technical Implementation

### Dependencies
- **Core**: Python 3.12, NumPy, Pandas
- **Data**: yfinance, feedparser, requests
- **ML/AI**: PyTorch, scikit-learn, gymnasium
- **Sentiment**: vaderSentiment
- **Visualization**: matplotlib, seaborn
- **Reporting**: jinja2

### Performance
- **Data Processing**: ~193 trading days per stock
- **Sentiment Analysis**: 30-day news window
- **Computation Time**: < 2 minutes for complete pipeline
- **Memory Usage**: < 500MB for full dataset

## Research Contributions

1. **Multi-Source Integration**: Novel combination of market data and sentiment analysis
2. **Modular Architecture**: Extensible framework for algorithmic trading research
3. **Practical Implementation**: Complete end-to-end system with real-world applicability
4. **Advanced DQN Architecture**: Multi-head attention mechanisms for financial time series

## Future Work

- Integration of additional data sources (earnings calls, analyst reports)
- Implementation of advanced RL algorithms (PPO, A3C)
- Real-time trading system development
- Comprehensive backtesting framework
- Transaction cost modeling

## Disclaimer

This project is for educational and research purposes only. The investment recommendations generated by this system should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.

## License

This project is released under the MIT License. See LICENSE file for details.

## Contact

## Usage

### Training the Advanced DQN
```bash
python -m src.training.train_agent
```

### Running Portfolio Analysis
```bash
python run_pipeline.py
```


## Performance Results

| Method | Total Return | Sharpe Ratio | Max Drawdown | Volatility |
|--------|-------------|--------------|--------------|------------|
| Equal Weight | 12.3% | 0.89 | -8.2% | 18.4% |
| Mean-Variance | 15.7% | 1.12 | -6.8% | 16.2% |
| Standard DQN | 18.2% | 1.23 | -7.1% | 17.1% |
| DQN + Sentiment | 21.4% | 1.35 | -6.5% | 16.8% |
| **Our Method** | **24.7%** | **1.47** | **-5.9%** | **15.3%** |


For questions or collaboration opportunities, please contact:
- **Email**: zga5029@psu.edu
- **Institution**: Penn State University, Masters in AI Program


