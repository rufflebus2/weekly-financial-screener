import os
import json
import requests
import yfinance as yf
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Loads variables from a local .env file when running on your Mac.
# In GitHub Actions, this safely does nothing and uses your repo Secrets instead.
load_dotenv()

# --- Configuration & Secrets (Pulled from Environment) ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Watchlist to screen weekly
WATCHLIST = ["GLD", "VOO", "VXUS", "SMH", "BTC-USD"]


# --- Structured Output Schema ---
class AssetSignal(BaseModel):
    ticker: str
    action: str = Field(description="Must be BUY, SELL, or HOLD")
    confidence: int = Field(description="Confidence score from 0 to 100")
    entry_price: str = Field(description="Suggested entry price or range")
    target_price: str = Field(description="Take-profit target price")
    stop_loss: str = Field(description="Stop-loss level")
    reasoning: str = Field(description="Concise 1-2 sentence technical and macro thesis")


# --- Market Data & AI Analysis ---
def fetch_market_context(ticker: str) -> str:
    asset = yf.Ticker(ticker)
    hist = asset.history(period="1mo")
    if hist.empty:
        return f"No recent price data found for {ticker}."

    latest_close = round(hist["Close"].iloc[-1], 2)
    high_1m = round(hist["High"].max(), 2)
    low_1m = round(hist["Low"].min(), 2)

    news_items = asset.news[:3] if asset.news else []
    headlines = [
        item.get("content", {}).get("title", "") or item.get("title", "")
        for item in news_items
    ]
    headlines_str = "; ".join([h for h in headlines if h]) or "No major headlines."

    return (
        f"Ticker: {ticker}\n"
        f"Latest Close: {latest_close}\n"
        f"1-Month High: {high_1m} | 1-Month Low: {low_1m}\n"
        f"Recent Headlines: {headlines_str}"
    )


def analyze_asset(client: genai.Client, ticker: str) -> AssetSignal:
    market_data = fetch_market_context(ticker)
    prompt = (
        "Analyze the following weekly asset data and generate a structured "
        "swing/position trading signal for manual execution:\n\n"
        f"{market_data}"
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AssetSignal,
            temperature=0.1,
        ),
    )
    return response.parsed


# --- Telegram Notification ---
def send_telegram_alert(signal: AssetSignal):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Skipping Telegram alert: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return

    text = (
        f"🚨 *{signal.ticker} Signal: {signal.action}* ({signal.confidence}% Conf)\n\n"
        f"• *Entry:* `{signal.entry_price}`\n"
        f"• *Target:* `{signal.target_price}`\n"
        f"• *Stop Loss:* `{signal.stop_loss}`\n\n"
        f"💡 *Thesis:* {signal.reasoning}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Sent Telegram alert for {signal.ticker}")
    except Exception as e:
        print(f"Failed to send Telegram alert for {signal.ticker}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID environment variable is missing.")

    # Automatically uses GOOGLE_APPLICATION_CREDENTIALS from your environment
    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT_ID,
        location=GCP_LOCATION,
    )

    for ticker in WATCHLIST:
        try:
            print(f"Analyzing {ticker}...")
            signal = analyze_asset(client, ticker)
            print(json.dumps(signal.model_dump(), indent=2))
            send_telegram_alert(signal)
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")