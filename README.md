# LLM for Real-Time Financial Forecasting

This Python package combines Large Language Models (LLMs) with traditional technical analysis for more accurate and interpretable financial forecasting.

## Overview

The `FinancialLLMForecaster` framework balances prediction performance with interpretability, making it suitable for financial professionals who need to understand the reasoning behind automated forecasts.

## Features

- **LLM-Based Sentiment Analysis**: Uses FinBERT to analyze sentiment from financial news
- **Multi-Factor Forecasting**: Combines technical indicators with sentiment analysis
- **Interpretability Layer**: Generates human-readable explanations for every forecast
- **Visualization**: Creates annotated charts showing predictions with contributing factors
- **Real-Time Data**: Works with the latest market data and news for up-to-date forecasts

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn torch transformers yfinance
```

## Usage

```python
# Basic usage with default settings
python financial_forecasting.py

# Or import in your own code
from financial_forecasting import FinancialLLMForecaster

# Create a forecaster for a specific ticker
forecaster = FinancialLLMForecaster("MSFT", lookback_days=30, forecast_days=5)

# Run the complete analysis
results = forecaster.run_complete_analysis()

# View the explanation
print(forecaster.explanation_text)
```

## Output

The script generates:
1. A visualization saved as a PNG file
2. Detailed results saved as a JSON file
3. A printed summary in the console
4. Interpretable explanation of the forecast factors

## Requirements

- Python 3.7+
- Dependencies listed in the installation section

## Citation

If you use this code in your research or project, please cite it as follows:

```
@software{financial_llm_forecaster_2025,
  author = {Nikolaos Roufas},
  title = {LLM for Real-Time Financial Forecasting},
  year = {2025},
  month = {March},
  url = {https://github.com/nikolasroufas/financial-llm-forecaster}
}
```
