import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from datetime import timedelta
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import io
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

app = Flask(__name__)
CORS(app)

def fetch_data(ticker, start_date='2024-01-01', end_date='2025-04-02'):
    """Fetch historical stock data."""
    df = yf.Ticker(ticker).history(period='1d', start=start_date, end=end_date)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def prepare_data(df):
    """Prepare data for forecasting by calculating returns and direction."""
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = (df['Return'] > 0).astype(int)
    df.dropna(inplace=True)
    df.index = df.index.tz_localize(None)
    return df

def fetch_headline_sentiment(query='gold'):
    """Fetch sentiment from news headlines."""
    try:
        url = f"https://news.google.com/rss/search?q={query}+price"
        resp = requests.get(url)
        from xml.etree import ElementTree as ET
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")

        headlines = [item.find('title').text for item in items[:5]]  # Top 5 headlines
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(h)['compound'] for h in headlines]

        avg_score = sum(scores) / len(scores) if scores else 0
        sentiment = (
            "Positive" if avg_score > 0.2 else
            "Negative" if avg_score < -0.2 else
            "Neutral"
        )

        return {
            "average_sentiment_score": round(avg_score, 3),
            "sentiment_summary": sentiment,
            "sample_headlines": headlines
        }

    except Exception as e:
        return {
            "average_sentiment_score": 0,
            "sentiment_summary": "Unavailable",
            "sample_headlines": [],
            "error": str(e)
        }

def combine_forecasts(trend_prophet, trend_xgb, last_close, sentiment_data):
    # If both models agree, use that recommendation
    if (trend_prophet > last_close and trend_xgb > last_close):
        recommendation = "Buy"
    elif (trend_prophet < last_close and trend_xgb < last_close):
        recommendation = "Hold/Sell"
    else:
        # If models defer, calculate the average trend
        combined_trend = (trend_prophet + trend_xgb) / 2
        if combined_trend > last_close:
            recommendation = "Buy"
        elif combined_trend < last_close:
            recommendation = "Hold/Sell"
        else:
            recommendation = "Hold"

    # Adjust the recommendation based on sentiment
    if sentiment_data['sentiment_summary'] == "Negative" and recommendation == "Buy":
        recommendation += " (but sentiment is negative)"
    elif sentiment_data['sentiment_summary'] == "Positive" and recommendation == "Hold/Sell":
        recommendation += " (but sentiment is positive)"

    return recommendation

def forecast_ticker(ticker='GC=F', days=7):
    """Forecast stock prices using Prophet and XGBoost models with confidence estimation."""
    try:
        df = prepare_data(fetch_data(ticker))

        # --- Prophet Forecast ---
        ts_data = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        ts_data['ds'] = ts_data['ds'].dt.tz_localize(None)
        prophet_model = Prophet()
        prophet_model.fit(ts_data)
        future = prophet_model.make_future_dataframe(periods=days)
        forecast_prophet = prophet_model.predict(future)
        forecast_p = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)

        prophet_ci_range = float((forecast_p['yhat_upper'] - forecast_p['yhat_lower']).mean())

        # --- XGBoost Forecast ---
        xgb_df = df.copy()
        xgb_df['Close_Lag_1'] = xgb_df['Close'].shift(1)
        xgb_df['Close_Lag_2'] = xgb_df['Close'].shift(2)
        xgb_df['Close_Lag_3'] = xgb_df['Close'].shift(3)
        xgb_df.dropna(inplace=True)

        features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']
        X = xgb_df[features]
        y = xgb_df['Close']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_xgb = XGBRegressor()
        model_xgb.fit(X_scaled, y)

        recent = xgb_df.iloc[-1][features].values.tolist()
        xgb_preds = []
        dates = []
        current_date = df.index[-1]

        for _ in range(days):
            recent_df = pd.DataFrame([recent], columns=features)
            recent_scaled = scaler.transform(recent_df)
            next_pred = float(model_xgb.predict(recent_scaled)[0])
            xgb_preds.append(next_pred)
            current_date += pd.Timedelta(days=1)
            dates.append(current_date)
            recent = [next_pred] + recent[:-1]

        forecast_xgb = pd.DataFrame({'ds': dates, 'yhat_xgb': xgb_preds})
        xgb_std_dev = float(np.std(xgb_preds))

        # --- Recommendation Logic ---
        last_close = df['Close'].iloc[-1]
        trend_prophet = float(forecast_p['yhat'].mean())
        trend_xgb = float(forecast_xgb['yhat_xgb'].mean())
        avg_forecast = (trend_prophet + trend_xgb) / 2

        sentiment_data = fetch_headline_sentiment(query=ticker)
        recommendation = combine_forecasts(trend_prophet, trend_xgb, last_close, sentiment_data)

        # --- Support and Resistance ---
        last_10 = df['Close'].tail(10)
        support = float(last_10.min())
        resistance = float(last_10.max())

        info = yf.Ticker(ticker).info
        name = info.get("shortName", ticker)
        sentiment_data = fetch_headline_sentiment(query=name)

        # --- Summary ---
        trend_direction = "increase" if avg_forecast > last_close else "decrease"
        trend_strength = "slightly" if abs(avg_forecast - last_close) / last_close < 0.01 else "significantly"
        summary_sentiment = sentiment_data.get('sentiment_summary', 'Unknown').lower()

        summary = (
            f"{name} prices are expected to {trend_strength} {trend_direction} over the next {days} days. "
            f"The Prophet model predicts an average price of ${round(trend_prophet, 2)}, with a confidence interval of ±${round(prophet_ci_range, 2)}. "
            f"XGBoost forecasts an average price of ${round(trend_xgb, 2)} with a prediction standard deviation of ±${round(xgb_std_dev, 2)}. "
            f"Current price is ${round(last_close, 2)}. Sentiment from recent headlines is {summary_sentiment}. "
            f"Support is near ${round(support, 2)}, resistance around ${round(resistance, 2)}. "
            f"Recommended action: {recommendation}."
        )

        return {
            "ticker": ticker,
            "name": name,
            "prediction_prophet": forecast_p.to_dict(orient='records'),
            "prediction_xgb": forecast_xgb.to_dict(orient='records'),
            "prophet_confidence_range": round(prophet_ci_range, 2),
            "xgb_prediction_std": round(xgb_std_dev, 2),
            "recommendation": recommendation,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "news_sentiment": sentiment_data,
            "summary": summary
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e)
        }


def generate_forecast_plot(ticker='GC=F', days=7):
    """Generate and return a forecast plot with Prophet and XGBoost confidence intervals."""
    df = prepare_data(fetch_data(ticker))
    df_reset = df.reset_index()

    # --- Prophet Forecast ---
    ts_data = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    ts_data['ds'] = ts_data['ds'].dt.tz_localize(None)
    prophet_model = Prophet().fit(ts_data)
    future = prophet_model.make_future_dataframe(periods=days)
    forecast_prophet = prophet_model.predict(future)
    forecast_p = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)

    # --- XGBoost Forecast ---
    df_xgb = df.copy()
    df_xgb['ds'] = df_xgb.index
    df_xgb = df_xgb[['ds', 'Close']].reset_index(drop=True)
    df_xgb['dayofweek'] = df_xgb['ds'].dt.dayofweek
    df_xgb['day'] = df_xgb['ds'].dt.day
    df_xgb['month'] = df_xgb['ds'].dt.month
    df_xgb['year'] = df_xgb['ds'].dt.year

    X = df_xgb[['dayofweek', 'day', 'month', 'year']]
    y = df_xgb['Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb_model = XGBRegressor()
    xgb_model.fit(X_scaled, y)

    future_dates = [df_xgb['ds'].max() + timedelta(days=i+1) for i in range(days)]
    future_features = pd.DataFrame({
        'dayofweek': [d.weekday() for d in future_dates],
        'day': [d.day for d in future_dates],
        'month': [d.month for d in future_dates],
        'year': [d.year for d in future_dates]
    })

    future_scaled = scaler.transform(future_features)
    yhat_xgb = xgb_model.predict(future_scaled)
    xgb_std = float(np.std(yhat_xgb))

    forecast_xgb = pd.DataFrame({
        'ds': future_dates,
        'yhat_xgb': yhat_xgb,
        'yhat_xgb_lower': yhat_xgb - xgb_std,
        'yhat_xgb_upper': yhat_xgb + xgb_std
    })

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(df_reset['Date'], df_reset['Close'], label='Historical Close', color='black')

    # Prophet
    plt.plot(forecast_p['ds'], forecast_p['yhat'], label='Prophet Forecast', color='blue')
    plt.fill_between(forecast_p['ds'], forecast_p['yhat_lower'], forecast_p['yhat_upper'],
                     color='blue', alpha=0.2, label='Prophet Confidence Interval')

    # XGBoost
    plt.plot(forecast_xgb['ds'], forecast_xgb['yhat_xgb'], label='XGBoost Forecast', color='green')
    plt.fill_between(forecast_xgb['ds'], forecast_xgb['yhat_xgb_lower'], forecast_xgb['yhat_xgb_upper'],
                     color='green', alpha=0.2, label='XGBoost ±1 Std Dev')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Price Forecast - Prophet vs XGBoost')
    plt.legend()

    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', default='GC=F')
    days = request.args.get('days', default=7, type=int)
    if days < 1: days = 1
    if days > 30: days = 30
    result = forecast_ticker(ticker, days)
    return jsonify(result)

@app.route('/plot', methods=['GET'])
def plot():
    ticker = request.args.get('ticker', default='GC=F')
    days = request.args.get('days', default=7, type=int)
    if days < 1: days = 1
    if days > 30: days = 30

    try:
        plot_buffer = generate_forecast_plot(ticker, days)
        return send_file(plot_buffer, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "App is running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

