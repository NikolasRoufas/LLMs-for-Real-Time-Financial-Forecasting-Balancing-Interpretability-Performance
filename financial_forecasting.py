"""
LLM for Real-Time Financial Forecasting
Author: Nikolaos Roufas
Date: March 22, 2025
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yfinance as yf
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialLLMForecaster:
    """
    A class that combines LLM-based sentiment analysis with traditional time series methods
    for real-time financial forecasting with interpretable outputs.
    """
    
    def __init__(self, ticker_symbol, lookback_days=30, forecast_days=5):
        """
        Initialize the forecaster with a specific stock ticker
        
        Args:
            ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL')
            lookback_days (int): Number of days to look back for historical data
            forecast_days (int): Number of days to forecast
        """
        self.ticker = ticker_symbol
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.sentiment_model = None
        self.tokenizer = None
        self.scaler = MinMaxScaler()
        self.historical_data = None
        self.news_data = None
        self.predictions = None
        self.explanation = {}
        
        logger.info(f"Initializing FinancialLLMForecaster for {ticker_symbol}")
    
    def _ensure_float(self, value):
        """
        Safely convert any value (including numpy arrays) to a Python float
        """
        if hasattr(value, 'item'):
            return value.item()  # Convert numpy scalar to native Python type
        try:
            return float(value)
        except:
            return 0.0  # Fallback
        
    def load_sentiment_model(self, model_name="ProsusAI/finbert"):
        """
        Load the language model for financial sentiment analysis
        
        Args:
            model_name (str): The name of the pre-trained model to use
        """
        logger.info(f"Loading sentiment model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            raise
    
    def fetch_historical_data(self):
        """
        Fetch historical price data for the ticker
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        logger.info(f"Fetching historical data for {self.ticker} from {start_date.date()} to {end_date.date()}")
        
        try:
            self.historical_data = yf.download(
                self.ticker, 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            if self.historical_data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
                
            # Calculate additional technical indicators
            self.historical_data['MA_5'] = self.historical_data['Close'].rolling(window=5).mean()
            self.historical_data['MA_20'] = self.historical_data['Close'].rolling(window=20).mean()
            self.historical_data['RSI'] = self._calculate_rsi(self.historical_data['Close'], 14)
            self.historical_data.dropna(inplace=True)
            
            logger.info(f"Successfully fetched {len(self.historical_data)} days of historical data")
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def _calculate_rsi(self, prices, window=14):
        """
        Calculate the Relative Strength Index (RSI) for interpretability
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def fetch_news(self, limit=20):
        """
        Fetch recent news articles related to the ticker symbol
        
        In a real-world implementation, this would connect to a news API.
        Here we'll simulate some news data.
        """
        logger.info(f"Fetching recent news for {self.ticker}")
        
        # Simulated news data - in a real implementation, fetch from a news API
        self.news_data = [
            {"title": f"Quarterly Results for {self.ticker}", 
             "content": f"{self.ticker} announced better than expected quarterly results with revenue growth of 15%.",
             "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")},
            {"title": f"New Product Launch from {self.ticker}", 
             "content": f"{self.ticker} is planning to launch new products next month, which could impact future growth.",
             "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")},
            {"title": "Market Analysis", 
             "content": f"Analysts predict a bullish trend for {self.ticker} in the coming weeks.",
             "date": datetime.now().strftime("%Y-%m-%d")}
        ]
        
        logger.info(f"Fetched {len(self.news_data)} news articles")
        
    def analyze_sentiment(self):
        """
        Analyze the sentiment of news articles using the loaded LLM
        """
        if not self.news_data or not self.sentiment_model:
            logger.error("News data or sentiment model not available")
            return
        
        logger.info("Analyzing news sentiment")
        
        sentiment_scores = []
        
        for article in self.news_data:
            # Combine title and content for analysis
            text = f"{article['title']}. {article['content']}"
            
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                
            # Get sentiment probabilities (positive, negative, neutral)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            sentiment_label = torch.argmax(probs, dim=1).item()
            
            # Map to sentiment labels (model specific, assuming 0=negative, 1=neutral, 2=positive)
            sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_mapping.get(sentiment_label, "neutral")
            
            sentiment_scores.append({
                "article": article['title'],
                "date": article['date'],
                "sentiment": sentiment,
                "probabilities": {
                    "negative": probs[0][0].item(),
                    "neutral": probs[0][1].item(),
                    "positive": probs[0][2].item()
                }
            })
        
        # Store for interpretability
        self.sentiment_analysis = sentiment_scores
        avg_sentiment = sum([1 if s['sentiment']=="positive" else (-1 if s['sentiment']=="negative" else 0) 
                             for s in sentiment_scores]) / len(sentiment_scores)
        
        logger.info(f"Sentiment analysis complete. Average sentiment: {avg_sentiment:.2f}")
        
        # Add to explanation
        self.explanation['sentiment'] = {
            'average_sentiment': avg_sentiment,
            'article_count': len(sentiment_scores),
            'sentiment_distribution': {
                'positive': sum(1 for s in sentiment_scores if s['sentiment']=="positive"),
                'neutral': sum(1 for s in sentiment_scores if s['sentiment']=="neutral"),
                'negative': sum(1 for s in sentiment_scores if s['sentiment']=="negative")
            }
        }
        
        return sentiment_scores
    
    def prepare_features(self):
        """
        Prepare features for the forecasting model, combining price data and sentiment
        """
        if self.historical_data is None:
            logger.error("Historical data not available")
            return
        
        logger.info("Preparing features for forecasting")
        
        # Create a copy of the historical data for feature engineering
        features_df = self.historical_data.copy()
        
        # Add technical indicators
        features_df['Price_Change'] = features_df['Close'].pct_change()
        features_df['Volume_Change'] = features_df['Volume'].pct_change()
        features_df['MA_Ratio'] = features_df['MA_5'] / features_df['MA_20']
        
        # Add sentiment features if available
        if hasattr(self, 'sentiment_analysis'):
            # Create a dictionary to map dates to sentiment scores
            sentiment_by_date = {}
            for article in self.sentiment_analysis:
                date = article['date']
                sentiment_score = 1 if article['sentiment']=="positive" else (-1 if article['sentiment']=="negative" else 0)
                
                if date in sentiment_by_date:
                    sentiment_by_date[date].append(sentiment_score)
                else:
                    sentiment_by_date[date] = [sentiment_score]
            
            # Calculate average sentiment score for each date
            for date in sentiment_by_date:
                sentiment_by_date[date] = sum(sentiment_by_date[date]) / len(sentiment_by_date[date])
            
            # Add sentiment scores to features_df
            features_df['Sentiment'] = 0
            for date, sentiment in sentiment_by_date.items():
                if date in features_df.index:
                    features_df.at[date, 'Sentiment'] = sentiment
        
        # Drop any rows with NaN values
        features_df.dropna(inplace=True)
        
        # Select features for the model
        self.feature_columns = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'Price_Change', 'Volume_Change', 'MA_Ratio']
        if 'Sentiment' in features_df.columns:
            self.feature_columns.append('Sentiment')
        
        # Scale the features
        self.features = features_df[self.feature_columns].values
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        logger.info(f"Features prepared: {self.feature_columns}")
        
        return self.features_scaled
    
    def train_forecast_model(self, model_type='hybrid'):
        """
        Train the forecasting model
        
        Args:
            model_type (str): Type of model to use ('statistical', 'ml', or 'hybrid')
        """
        logger.info(f"Training forecast model: {model_type}")
        
        # Calculate trend based on available days
        close_prices = self.historical_data['Close'].values
        days_to_look_back = min(6, len(close_prices))
        if days_to_look_back <= 1:
            recent_trend = 0  # Not enough data for trend
        else:
            recent_trend = (self._ensure_float(close_prices[-1]) - self._ensure_float(close_prices[-days_to_look_back])) / self._ensure_float(close_prices[-days_to_look_back])
        
        # Add technical indicators
        rsi = self._ensure_float(self.historical_data['RSI'].iloc[-1])
        ma_cross = self._ensure_float(self.historical_data['MA_5'].iloc[-1]) > self._ensure_float(self.historical_data['MA_20'].iloc[-1])
        
        # Get sentiment influence
        sentiment_factor = 0
        if hasattr(self, 'explanation') and 'sentiment' in self.explanation:
            sentiment_factor = self.explanation['sentiment']['average_sentiment'] * 0.02  # 2% influence
        
        # Combine factors for hybrid prediction
        trend_prediction = recent_trend
        tech_prediction = 0.01 if ma_cross else -0.01  # 1% influence from moving average cross
        rsi_prediction = -0.01 if rsi > 70 else (0.01 if rsi < 30 else 0)  # 1% influence from RSI
        
        # Store factors for interpretability
        self.explanation['forecast_factors'] = {
            'recent_trend': recent_trend,
            'ma_cross': 'bullish' if ma_cross else 'bearish',
            'rsi': rsi,
            'sentiment_factor': sentiment_factor
        }
        
        # Combine for final prediction
        combined_factor = trend_prediction + tech_prediction + rsi_prediction + sentiment_factor
        
        # Generate forecasts
        last_price = self._ensure_float(close_prices[-1])
        forecasts = []
        
        for i in range(self.forecast_days):
            # Compound the effect for multi-day forecasts
            next_price = last_price * (1 + combined_factor)
            forecasts.append(next_price)
            last_price = next_price
        
        self.predictions = forecasts
        
        logger.info(f"Forecast complete. Predicted {self.forecast_days}-day change: {(forecasts[-1]/self._ensure_float(close_prices[-1])-1)*100:.2f}%")
        
        return forecasts
    
    def generate_explanation(self):
        """
        Generate a human-readable explanation of the forecast
        """
        if not hasattr(self, 'explanation') or not self.predictions:
            logger.error("No forecast data available for explanation")
            return
        
        logger.info("Generating interpretable explanation of forecast")
        
        factors = self.explanation['forecast_factors']
        sentiment = self.explanation['sentiment']
        
        # Ensure values are Python native types
        last_close = self._ensure_float(self.historical_data['Close'].iloc[-1])
        final_prediction = self._ensure_float(self.predictions[-1])
        price_change = (final_prediction/last_close-1)*100
        
        # Create explanation text
        explanation_text = f"""
        # Financial Forecast Explanation for {self.ticker}
        
        ## Summary
        The model predicts a {'positive' if final_prediction > last_close else 'negative'} trend 
        over the next {self.forecast_days} days, with a projected price change of 
        {price_change:.2f}%.
        
        ## Key Factors
        
        1. **Recent Price Trend:** {factors['recent_trend']*100:.2f}% over last 5 days
        2. **Technical Indicators:**
        - Moving Average: {factors['ma_cross']} (5-day MA {'above' if factors['ma_cross']=='bullish' else 'below'} 20-day MA)
        - RSI: {factors['rsi']:.2f} ({self._interpret_rsi(factors['rsi'])})
        3. **News Sentiment Analysis:**
        - Overall sentiment: {self._sentiment_to_text(sentiment['average_sentiment'])}
        - Based on {sentiment['article_count']} recent articles
        - Distribution: {sentiment['sentiment_distribution']['positive']} positive, 
            {sentiment['sentiment_distribution']['neutral']} neutral, 
            {sentiment['sentiment_distribution']['negative']} negative
        
        ## Confidence Assessment
        
        The model's confidence is {'high' if abs(factors['recent_trend']) > 0.05 and abs(sentiment['average_sentiment']) > 0.5 else 'moderate' if abs(factors['recent_trend']) > 0.02 or abs(sentiment['average_sentiment']) > 0.3 else 'low'} 
        based on the alignment between technical indicators and sentiment analysis.
        
        ## Limitations
        
        This forecast does not account for unexpected news events, broader market movements, 
        or macroeconomic changes that may occur during the forecast period.
        """
        
        self.explanation_text = explanation_text
        logger.info("Explanation generated successfully")
        
        return explanation_text
    
    def _interpret_rsi(self, rsi_value):
        """Helper to interpret RSI values"""
        if rsi_value > 70:
            return "potentially overbought"
        elif rsi_value < 30:
            return "potentially oversold"
        else:
            return "neutral"
    
    def _sentiment_to_text(self, sentiment_value):
        """Helper to convert sentiment numerical values to text"""
        if sentiment_value > 0.5:
            return "strongly positive"
        elif sentiment_value > 0.1:
            return "positive"
        elif sentiment_value < -0.5:
            return "strongly negative"
        elif sentiment_value < -0.1:
            return "negative"
        else:
            return "neutral"
            
    def visualize_forecast(self):
        """
        Create a visualization of the historical data and forecast
        """
        if self.historical_data is None or self.predictions is None:
            logger.error("Historical data or predictions not available")
            return
        
        logger.info("Creating forecast visualization")
        
        # Create a date range for the forecast
        last_date = self.historical_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=self.forecast_days)
        
        # Create a dataframe for plotting
        plot_df = pd.DataFrame(index=forecast_dates, data={'Forecast': self.predictions})
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(self.historical_data.index, self.historical_data['Close'], label='Historical')
        plt.plot(plot_df.index, plot_df['Forecast'], label='Forecast', linestyle='--')
        
        # Add moving averages
        plt.plot(self.historical_data.index, self.historical_data['MA_5'], label='5-Day MA', alpha=0.7)
        plt.plot(self.historical_data.index, self.historical_data['MA_20'], label='20-Day MA', alpha=0.7)
        
        # Add explanatory notes for interpretability
        if hasattr(self, 'sentiment_analysis'):
            # Add sentiment annotations
            for article in self.sentiment_analysis:
                article_date = datetime.strptime(article['date'], '%Y-%m-%d')
                if article_date in self.historical_data.index:
                    price = self._ensure_float(self.historical_data.loc[article_date, 'Close'])
                    sentiment = article['sentiment']
                    color = 'green' if sentiment == 'positive' else ('red' if sentiment == 'negative' else 'gray')
                    plt.scatter(article_date, price, color=color, marker='o', s=50, alpha=0.7)
        
        plt.title(f"{self.ticker} Price Forecast with LLM-Enhanced Analysis")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Annotate the forecast
        last_close = self._ensure_float(self.historical_data['Close'].iloc[-1])
        final_prediction = self._ensure_float(self.predictions[-1])
        forecast_change = (final_prediction/last_close-1)*100
        
        plt.annotate(f"Forecast: {forecast_change:.2f}%", 
                    xy=(forecast_dates[-1], final_prediction),
                    xytext=(forecast_dates[-1], final_prediction * 1.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"{self.ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(filename)
        logger.info(f"Visualization saved as {filename}")
        
        plt.close()
        
    def run_complete_analysis(self):
        """
        Run the complete forecasting pipeline
        """
        start_time = time.time()
        logger.info(f"Starting complete analysis for {self.ticker}")
        
        # Load model
        self.load_sentiment_model()
        
        # Fetch data
        self.fetch_historical_data()
        self.fetch_news()
        
        # Analyze and forecast
        self.analyze_sentiment()
        self.prepare_features()
        self.train_forecast_model()
        
        # Generate interpretable outputs
        explanation = self.generate_explanation()
        self.visualize_forecast()
        
        # Ensure all values are Python native types
        last_close = self._ensure_float(self.historical_data['Close'].iloc[-1])
        
        # Convert all predictions to native Python types
        prediction_values = [self._ensure_float(p) for p in self.predictions]
        final_prediction = prediction_values[-1]
        forecast_change = (final_prediction/last_close-1)*100
        
        # Save results to file
        results = {
            "ticker": self.ticker,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "current_price": last_close,
            "forecast_prices": prediction_values,
            "forecast_change_percent": forecast_change,
            "explanation": self.explanation,
            "runtime_seconds": float(time.time() - start_time)
        }
        
        with open(f"{self.ticker}_forecast_results_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Complete analysis finished in {time.time() - start_time:.2f} seconds")
        
        return results


def main():
    # Example usage
    ticker = "AAPL"  # Change to desired ticker
    
    # Create forecaster instance
    forecaster = FinancialLLMForecaster(ticker, lookback_days=60, forecast_days=5)
    
    try:
        # Run the complete analysis pipeline
        results = forecaster.run_complete_analysis()
        
        # Print summary
        print("\n" + "="*50)
        print(f"Financial Forecast Summary for {ticker}")
        print("="*50)
        print(f"Current Price: ${results['current_price']:.2f}")
        print(f"5-Day Forecast: ${results['forecast_prices'][-1]:.2f} ({results['forecast_change_percent']:.2f}%)")
        print(f"Analysis Runtime: {results['runtime_seconds']:.2f} seconds")
        print("\nExplanation:")
        print(forecaster.explanation_text)
        print("\nDetailed results saved to JSON file.")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
