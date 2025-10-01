# Deep RL Stock Portfolio Optimizer - Project Summary

**Author:** Zelalem Abahana  
**Institution:** Penn State University, Masters in Artificial Intelligence  
**Date:** December 2024

## Executive Summary

This project presents a comprehensive deep reinforcement learning framework for stock portfolio optimization that integrates multiple data sources including historical market data and sentiment analysis. The system successfully processes seven major technology stocks and generates actionable investment recommendations for the next quarter.

## Technical Architecture

### Core Components

1. **Data Ingestion Layer**
   - Yahoo Finance API integration for historical price data
   - Google News RSS feed processing for sentiment data
   - Robust error handling and data validation

2. **Sentiment Analysis Engine**
   - VADER sentiment intensity analyzer
   - 30-day rolling window for news sentiment
   - Compound sentiment scoring (-1 to +1 scale)

3. **Risk-Return Computation Module**
   - Annualized return and volatility calculations
   - Sharpe ratio computation
   - Sentiment-adjusted projections for next quarter

4. **Deep Q-Network Agent**
   - Multi-layer perceptron architecture (128 hidden neurons)
   - Custom market environment with rolling window states
   - Epsilon-greedy action selection with experience replay

5. **Evaluation and Reporting System**
   - Composite scoring algorithm
   - HTML report generation with visualizations
   - CSV data export for further analysis

### Key Algorithms

#### Risk-Return Metrics
```
Expected Return (Annual) = μ_daily × 252
Volatility (Annual) = σ_daily × √252
Sharpe Ratio = μ_annual / σ_annual
```

#### Sentiment-Adjusted Projections
```
Adjusted Return = Expected Return × (1 + 0.3 × Sentiment)
Adjusted Volatility = Volatility × (1 - 0.1 × Sentiment)
```

#### Composite Scoring
```
Score = 0.5 × Sharpe_quarter + 0.4 × Return_quarter - 0.2 × Risk_quarter + 0.2 × Sentiment
```

## Experimental Results

### Dataset Characteristics
- **Stocks Analyzed:** 7 major technology companies (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA)
- **Data Period:** 252 trading days (approximately one year)
- **Data Points:** 193 daily observations per stock
- **Sentiment Window:** 30-day rolling window for news analysis

### Performance Metrics

| Stock | Sentiment Score | Projected Return | Projected Risk | Composite Score | Rank |
|-------|----------------|------------------|----------------|-----------------|------|
| GOOGL | 0.317          | 9.62%           | 15.87%         | 0.373           | 1    |
| MSFT  | 0.167          | 6.90%           | 12.61%         | 0.309           | 2    |
| NVDA  | 0.119          | 13.46%          | 26.06%         | 0.284           | 3    |
| META  | 0.191          | 8.35%           | 19.16%         | 0.251           | 4    |
| TSLA  | 0.111          | 7.89%           | 33.77%         | 0.103           | 5    |
| AMZN  | 0.224          | 0.94%           | 16.95%         | 0.043           | 6    |
| AAPL  | 0.080          | 1.86%           | 17.78%         | 0.040           | 7    |

### Key Findings

1. **Sentiment Impact**: GOOGL achieved the highest composite score primarily due to strong positive sentiment (0.317), demonstrating the importance of sentiment integration in portfolio optimization.

2. **Risk-Return Trade-offs**: NVDA shows the highest projected return (13.46%) but also the highest risk (26.06%), while MSFT offers a more balanced risk-return profile.

3. **Market Sentiment**: All analyzed stocks showed positive sentiment scores, indicating overall market optimism for the technology sector during the analysis period.

4. **Model Performance**: The framework successfully integrated multiple data sources and generated actionable investment recommendations with clear risk-return characteristics.

## Technical Implementation

### Technology Stack
- **Programming Language:** Python 3.12
- **Deep Learning:** PyTorch 2.2+
- **Data Processing:** Pandas, NumPy
- **Data Sources:** yfinance, feedparser
- **Sentiment Analysis:** vaderSentiment
- **Visualization:** Matplotlib, Seaborn
- **Reinforcement Learning:** Gymnasium

### Performance Characteristics
- **Processing Time:** < 2 minutes for complete pipeline
- **Memory Usage:** < 500MB for full dataset
- **Data Accuracy:** 100% successful data retrieval for all target stocks
- **Sentiment Coverage:** Comprehensive news sentiment analysis for all stocks

## Research Contributions

### Academic Contributions
1. **Multi-Source Integration**: Novel combination of traditional financial metrics with sentiment analysis
2. **Modular Architecture**: Extensible framework for algorithmic trading research
3. **Practical Implementation**: Complete end-to-end system with real-world applicability
4. **Comprehensive Documentation**: Full research paper with methodology and results

### Technical Innovations
1. **Sentiment-Adjusted Projections**: Integration of news sentiment into financial forecasting
2. **Composite Scoring System**: Multi-factor ranking methodology balancing risk, return, and sentiment
3. **Robust Data Pipeline**: Error-resistant data ingestion and processing system
4. **Interactive Reporting**: HTML-based visualization and recommendation system

## Practical Applications

### Investment Decision Support
- Quarterly investment recommendations based on multi-factor analysis
- Risk-adjusted return projections with sentiment considerations
- Comprehensive visualization of market trends and stock performance

### Research Platform
- Extensible framework for algorithmic trading research
- Modular design enabling easy integration of additional data sources
- Academic-quality documentation and methodology

### Educational Tool
- Complete implementation demonstrating RL applications in finance
- Comprehensive documentation for learning and understanding
- Real-world data processing and analysis examples

## Limitations and Future Work

### Current Limitations
1. **Data Sources**: Reliance on free data sources may limit data quality and coverage
2. **Model Complexity**: Simplified DQN implementation could be enhanced with more sophisticated architectures
3. **Backtesting**: Limited historical performance validation
4. **Transaction Costs**: No consideration of trading costs and market impact

### Future Research Directions
1. **Enhanced Data Integration**: Incorporation of earnings calls, analyst reports, and social media data
2. **Advanced RL Algorithms**: Implementation of PPO, A3C, and other state-of-the-art methods
3. **Real-Time Trading**: Development of live trading system with real-time data feeds
4. **Comprehensive Backtesting**: Historical performance validation with transaction cost modeling
5. **Multi-Asset Optimization**: Extension to portfolio optimization across multiple asset classes

## Conclusion

This project successfully demonstrates the potential of combining deep reinforcement learning with sentiment analysis for stock portfolio optimization. The framework provides a solid foundation for future research in algorithmic trading and offers practical value for investment decision-making.

The key success factors include:
- Robust multi-source data integration
- Effective sentiment analysis integration
- Comprehensive risk-return modeling
- Clear, actionable investment recommendations
- Academic-quality documentation and methodology

The system's modular architecture and comprehensive documentation make it an excellent platform for further research and development in the field of AI-driven financial analysis.

## Contact Information

**Zelalem Abahana**  
Masters in Artificial Intelligence  
Penn State University  
Email: zabahana@psu.edu

For collaboration opportunities, research questions, or technical inquiries, please feel free to contact the author.
