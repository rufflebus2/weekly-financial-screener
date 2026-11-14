"""
Weekly long-only signal screener.

Pipeline (runs every Monday shortly after the US market open via GitHub Actions):
  1. Fetch ~1 year of daily OHLCV from Yahoo Finance (yfinance >= 1.8.0 required).
  2. Compute technical indicators locally (trend, momentum, volatility, volume).
  3. Pull recent news (yfinance get_news, with Google News RSS fallback),
     filtered to the last 7 days so stale headlines cannot skew sentiment.
  4. Ask Gemini for a structured BUY/HOLD signal with a rubric-based confidence
     score, entry/target/stop levels, and a cited thesis.
  5. Validate the LLM output arithmetically (long-only logic, reward:risk,
     entry sanity) and retry once with feedback if it is wrong.
  6. Compute a risk-based position size (1% of account risked per trade).
  7. Send the alert to Telegram; notify on failures too.

Long-only cash account: SELL is never emitted. "HOLD" means "do not enter".
"""

import html
import json
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Literal, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Loads variables from a local .env file when running on your Mac.
# In GitHub Actions this safely does nothing and repo Secrets are used instead.
load_dotenv()

ET_TZ = ZoneInfo("America/New_York")

# --- Configuration & Secrets (pulled from environment) -------------------------
# GitHub Actions renders an undefined ${{ vars.X }} / missing secret as an EMPTY
# string, so plain os.getenv(name, default) returns "" (not the default) and
# float("") crashes. These helpers treat "" as unset and fall back safely.


def env_str(name: str, default: str = "") -> str:
    value = (os.getenv(name) or "").strip()
    return value or default


def env_float(name: str, default: float) -> float:
    value = (os.getenv(name) or "").strip().replace(",", "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Warning: {name}={value!r} is not a number; using default {default}.")
        return default


WATCHLIST = [
    t.strip().upper()
    for t in env_str("WATCHLIST", "NVDA").split(",")
    if t.strip()
]
GEMINI_MODEL = env_str("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Auth: Vertex AI (GCP service account) or Gemini API key --------------------
# Hong Kong (and other regions outside the AI Studio list) must use Vertex AI
# with a service account; the API-key path only works in supported regions.
# See https://ai.google.dev/gemini-api/docs/available-regions
GCP_PROJECT_ID = env_str("GCP_PROJECT_ID")
GCP_LOCATION = env_str("GCP_LOCATION", "us-central1")

# Position sizing (USD). ACCOUNT_SIZE=0 disables the sizing block in the alert.
ACCOUNT_SIZE = env_float("ACCOUNT_SIZE", 0.0)
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 1.0)  # % of account risked per trade
MAX_POSITION_PCT = env_float("MAX_POSITION_PCT", 25.0)     # max % of account in one position

MIN_REWARD_RISK = 1.5          # minimum (target-entry)/(entry-stop) for a BUY
MIN_BUY_CONFIDENCE = 60        # a BUY below this confidence is invalid
NEWS_WINDOW_DAYS = 7           # only headlines from the last N days count
MAX_NEWS_ITEMS = 8
FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"  # skip the market-open check (for manual tests)


# --- Structured output schema ---
class AssetSignal(BaseModel):
    ticker: str = Field(description="Ticker symbol analyzed")
    action: Literal["BUY", "HOLD"] = Field(
        description="BUY only if a long entry with reward:risk >= 1.5 is justified; otherwise HOLD. Never SELL (long-only cash account)."
    )
    confidence: int = Field(
        ge=0, le=100,
        description="Rubric: 80-100 multiple indicators aligned + supportive news; 60-79 moderately aligned; 0-59 mixed/conflicting (never pair with BUY).",
    )
    entry_price: float = Field(description="Suggested entry price, within 2% of the latest close")
    target_price: float = Field(description="Take-profit target, a realistic resistance level above entry")
    stop_loss: float = Field(description="Stop-loss below a real support level (swing low / EMA20 / ~1x ATR)")
    reasoning: str = Field(description="2-3 sentences citing the specific indicator values and headlines that drove the decision")


# --- Market data -----------------------------------------------------------

def fetch_history(ticker: str, retries: int = 3) -> pd.DataFrame:
    """Daily bars for ~1 year, with retries against Yahoo rate limits."""
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period="1y", interval="1d")
            if not hist.empty:
                return hist
        except Exception as e:
            print(f"yfinance attempt {attempt + 1} failed for {ticker}: {e}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch price history for {ticker} after {retries} attempts.")


def check_trading_day(hist: pd.DataFrame, ticker: str) -> bool:
    """Skip the run when the US market appears closed today (holiday) or data is stale.

    At run time (shortly after the US open on a scheduled Monday) today's daily
    bar should already exist. If the newest bar is several days old, today is
    most likely a market holiday (MLK, Presidents Day, Memorial Day, Labor Day).
    Set FORCE_RUN=1 to bypass this check for manual tests.
    """
    if FORCE_RUN:
        return True
    today = datetime.now(ET_TZ).date()
    last_bar = hist.index[-1].date()
    if (today - last_bar).days > 4:
        print(f"Data for {ticker} looks stale (last bar {last_bar}); aborting to avoid bad signals.")
        return False
    if last_bar != today:
        print(f"No bar for today ({today}) yet for {ticker}; likely a market holiday or data lag. Skipping run.")
        return False
    return True


def compute_indicators(hist: pd.DataFrame) -> dict:
    close = hist["Close"]
    latest = close.iloc[-1]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # RSI(14), Wilder smoothing
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).iloc[-1]

    # MACD(12, 26, 9)
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - signal_line).iloc[-1]

    # ATR(14), Wilder smoothing
    prev_close = close.shift(1)
    tr = pd.concat(
        [hist["High"] - hist["Low"],
         (hist["High"] - prev_close).abs(),
         (hist["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

    vol = hist["Volume"]
    vol_ratio = (vol.iloc[-1] / vol.tail(20).mean()) if vol.tail(20).mean() > 0 else None

    high_52w, low_52w = hist["High"].max(), hist["Low"].min()

    def ret(bars: int) -> Optional[float]:
        return round((latest / close.iloc[-1 - bars] - 1) * 100, 1) if len(close) > bars else None

    return {
        "latest_close": round(float(latest), 2),
        "ema20": round(float(ema20.iloc[-1]), 2),
        "ema50": round(float(ema50.iloc[-1]), 2),
        "ema200": round(float(ema200.iloc[-1]), 2) if len(close) >= 200 else None,
        "rsi14": round(float(rsi), 1),
        "macd_histogram": round(float(macd_hist), 3),
        "atr14": round(float(atr), 2),
        "atr_pct_of_price": round(float(atr / latest * 100), 2),
        "volume_vs_20d_avg": round(float(vol_ratio), 2) if vol_ratio is not None else None,
        "high_52w": round(float(high_52w), 2),
        "low_52w": round(float(low_52w), 2),
        "pct_below_52w_high": round(float((latest / high_52w - 1) * 100), 1),
        "return_1w_pct": ret(5),
        "return_1m_pct": ret(21),
        "return_3m_pct": ret(63),
        "data_as_of": str(hist.index[-1].date()),
    }


# --- News ------------------------------------------------------------------

def _parse_dt(value) -> Optional[datetime]:
    """Tolerant parser for yfinance pubDate (ISO or epoch) and RSS pubDate."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        pass
    try:
        return parsedate_to_datetime(str(value))
    except (ValueError, TypeError):
        return None


def fetch_yahoo_news(ticker: str) -> list[dict]:
    """yfinance >= 1.8.0 news (Ticker.news / the old /xhr/ncp route is dead since Oct 2026)."""
    items = []
    try:
        raw = yf.Ticker(ticker).get_news(count=12) or []
    except Exception as e:
        print(f"yfinance news failed for {ticker}: {e}")
        return items
    for article in raw:
        content = (article.get("content") or {}) if isinstance(article, dict) else {}
        title = content.get("title") or ""
        if not title:
            continue
        publisher = (content.get("provider") or {}).get("displayName") or ""
        items.append({"title": title, "publisher": publisher, "published": _parse_dt(content.get("pubDate"))})
    return items


def fetch_google_news_rss(ticker: str) -> list[dict]:
    """Fallback source: Google News RSS search for the ticker."""
    url = (
        "https://news.google.com/rss/search"
        f"?q={requests.utils.quote(ticker + ' stock')}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        resp = requests.get(
            url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; signal-bot/1.0)"}
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"Google News RSS failed for {ticker}: {e}")
        return []
    items = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "publisher": node.findtext("source") or "",
            "published": _parse_dt(node.findtext("pubDate")),
        })
    return items


def fetch_recent_news(ticker: str) -> list[dict]:
    """Headlines from the last NEWS_WINDOW_DAYS days, newest first, capped at MAX_NEWS_ITEMS."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_WINDOW_DAYS)
    items = fetch_yahoo_news(ticker) or fetch_google_news_rss(ticker)
    recent = [i for i in items if i["published"] is None or i["published"] >= cutoff]
    recent.sort(key=lambda i: i["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return recent[:MAX_NEWS_ITEMS]


# --- Gemini analysis --------------------------------------------------------

def build_prompt(ticker: str, ind: dict, news: list[dict]) -> str:
    news_lines = (
        "\n".join(
            f"- [{(i['published'] or datetime.now(timezone.utc)).strftime('%Y-%m-%d')}"
            f"{' | ' + i['publisher'] if i['publisher'] else ''}] {i['title']}"
            for i in news
        )
        or f"No significant headlines found in the last {NEWS_WINDOW_DAYS} days."
    )

    indicators_block = "\n".join(f"{k.replace('_', ' ')}: {v}" for k, v in ind.items())

    return (
        f"You are a disciplined swing-trading analyst. The client trades a LONG-ONLY cash "
        f"account: no short selling, no leverage, no options. Decide whether {ticker} offers "
        f"an attractive long entry for a 1-4 week swing horizon, using ONLY the data below. "
        f"Never invent numbers or headlines that are not in the data.\n\n"
        f"=== TECHNICAL SNAPSHOT (daily bars) ===\n{indicators_block}\n\n"
        f"=== NEWS (last {NEWS_WINDOW_DAYS} days) ===\n{news_lines}\n\n"
        f"=== DECISION RULES ===\n"
        f"1. action must be BUY only if technicals AND news support a long entry with "
        f"reward:risk >= {MIN_REWARD_RISK}. Otherwise HOLD (HOLD also covers 'avoid / no edge'). "
        f"NEVER output SELL.\n"
        f"2. For BUY: entry_price within 2% of the latest close; stop_loss below a real support "
        f"level (recent swing low, EMA20, or ~1x ATR14); target_price at a realistic resistance "
        f"level above entry.\n"
        f"3. Confidence rubric (0-100): 80-100 = multiple indicators aligned (trend, momentum, "
        f"volume) AND news supportive; 60-79 = moderately aligned with some conflicts; 0-59 = "
        f"mixed or conflicting (a BUY must never score below {MIN_BUY_CONFIDENCE}).\n"
        f"4. In reasoning, cite the specific indicator values and headline(s) that drove the "
        f"decision, and note the single biggest risk to the trade.\n"
        f"5. Some headlines may be loosely related or unrelated to {ticker} (syndicated "
        f"feeds are noisy); ignore any that do not concern {ticker} or the broader market."
    )


def validate_signal(signal: AssetSignal, latest_close: float) -> list[str]:
    """Arithmetic long-only validation of the LLM output."""
    problems = []
    if signal.action == "BUY":
        if not (signal.stop_loss < signal.entry_price < signal.target_price):
            problems.append("For BUY, require stop_loss < entry_price < target_price.")
        else:
            rr = (signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss)
            if rr < MIN_REWARD_RISK:
                problems.append(
                    f"Reward:risk is {rr:.2f} but the minimum is {MIN_REWARD_RISK}."
                )
        if abs(signal.entry_price / latest_close - 1) > 0.05:
            problems.append("Entry price is more than 5% away from the latest close.")
        if signal.confidence < MIN_BUY_CONFIDENCE:
            problems.append(f"A BUY signal must have confidence >= {MIN_BUY_CONFIDENCE}.")
    return problems


def analyze_asset(client: genai.Client, ticker: str, ind: dict, news: list[dict]) -> AssetSignal:
    base_prompt = build_prompt(ticker, ind, news)
    prompt = base_prompt
    signal = None
    problems = []

    for attempt in range(2):
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AssetSignal,
                temperature=0.1,
            ),
        )
        signal = response.parsed
        if signal is None:
            raise RuntimeError(f"Gemini returned no parsed output for {ticker} (attempt {attempt + 1}).")
        signal.ticker = ticker
        problems = validate_signal(signal, ind["latest_close"])
        if not problems:
            return signal
        print(f"Validation issues on attempt {attempt + 1} for {ticker}: {problems}")
        prompt = (
            base_prompt
            + "\n\nYour previous answer had these problems — fix them and answer again:\n- "
            + "\n- ".join(problems)
        )

    # Still invalid after the retry: never act on a broken BUY.
    signal.action = "HOLD"
    signal.reasoning = (
        f"Downgraded to HOLD because the model's BUY levels failed validation: "
        + "; ".join(problems)
    )
    return signal


# --- Position sizing --------------------------------------------------------

def compute_position_size(signal: AssetSignal) -> Optional[dict]:
    """Risk-based sizing: risk RISK_PER_TRADE_PCT of the account if the stop hits."""
    if ACCOUNT_SIZE <= 0 or signal.action != "BUY":
        return None
    risk_per_share = signal.entry_price - signal.stop_loss
    if risk_per_share <= 0:
        return None
    risk_budget = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    shares = int(risk_budget / risk_per_share)
    shares = min(shares, int(ACCOUNT_SIZE * MAX_POSITION_PCT / 100 / signal.entry_price))
    if shares < 1:
        return {"shares": 0, "note": "Risk budget too small for even one share at these levels."}
    return {
        "shares": shares,
        "cost": round(shares * signal.entry_price, 2),
        "risk_amount": round(shares * risk_per_share, 2),
        "risk_pct": round(shares * risk_per_share / ACCOUNT_SIZE * 100, 2),
    }


# --- Telegram ----------------------------------------------------------------

def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Skipping Telegram alert: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        print(text)
        return
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")


def format_alert(signal: AssetSignal, ind: dict, sizing: Optional[dict]) -> str:
    rr = (
        (signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss)
        if signal.entry_price != signal.stop_loss
        else 0
    )
    icon = "📈" if signal.action == "BUY" else "⏸"
    lines = [
        f"{icon} <b>{signal.ticker} — {signal.action}</b> "
        f"({signal.confidence}% confidence)",
        f"Entry: <b>{signal.entry_price:,.2f}</b> | Target: <b>{signal.target_price:,.2f}</b> | "
        f"Stop: <b>{signal.stop_loss:,.2f}</b> | R:R ≈ {rr:.1f}",
        f"Data as of {ind['data_as_of']} close (RSI {ind['rsi14']}, ATR {ind['atr14']})",
    ]
    if sizing:
        if sizing.get("shares"):
            lines.append(
                f"Suggested size ({RISK_PER_TRADE_PCT:g}% risk on ${ACCOUNT_SIZE:,.0f}): "
                f"<b>{sizing['shares']} shares</b> ≈ ${sizing['cost']:,.0f} "
                f"(risks ${sizing['risk_amount']:,.0f} = {sizing['risk_pct']:.1f}% if stop hits)"
            )
        else:
            lines.append(f"Sizing: {sizing.get('note', '')}")
    lines.append("")
    lines.append(f"<i>{html.escape(signal.reasoning)}</i>")
    return "\n".join(lines)


# --- Main execution -----------------------------------------------------------

def build_client() -> genai.Client:
    """Create the Gemini client.

    Auth path 1 — Vertex AI (required in regions where the Gemini API key /
    AI Studio is unavailable, e.g. Hong Kong):
      - GCP_PROJECT_ID: your GCP project id
      - Service account credentials, one of:
        * GOOGLE_APPLICATION_CREDENTIALS: path to the service-account JSON file
          (local .env workflow), or
        * GOOGLE_APPLICATION_CREDENTIALS_JSON: the raw JSON content of the key
          (GitHub Actions secret) — written to a temp file and used via ADC.
      - The service account needs the "Vertex AI User" role.

    Auth path 2 — Gemini API key (supported regions only): GEMINI_API_KEY.
    """
    if GCP_PROJECT_ID:
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
            # GitHub Actions: secret holds the raw key JSON; ADC needs a file path.
            fd, creds_path = tempfile.mkstemp(suffix=".json")
            with os.fdopen(fd, "w") as f:
                f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        print(f"Auth: Vertex AI, project {GCP_PROJECT_ID}, location {GCP_LOCATION}")
        return genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)

    if GEMINI_API_KEY:
        print("Auth: Gemini API key (Vertex AI not configured).")
        return genai.Client(api_key=GEMINI_API_KEY)

    raise RuntimeError(
        "No Gemini credentials found. Either set GCP_PROJECT_ID + a service account "
        "(GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS_JSON) for "
        "Vertex AI — required in Hong Kong — or set GEMINI_API_KEY if you are in a "
        "supported region. See https://ai.google.dev/gemini-api/docs/available-regions"
    )


def main() -> int:
    client = build_client()

    sent, failed = 0, 0
    for ticker in WATCHLIST:
        try:
            print(f"Analyzing {ticker}...")
            hist = fetch_history(ticker)
            if not check_trading_day(hist, ticker):
                continue

            ind = compute_indicators(hist)
            news = fetch_recent_news(ticker)
            print(f"  indicators: {ind['latest_close']} close, RSI {ind['rsi14']}, {len(news)} recent headlines")

            signal = analyze_asset(client, ticker, ind, news)
            sizing = compute_position_size(signal)
            print(json.dumps(signal.model_dump(), indent=2))

            send_telegram(format_alert(signal, ind, sizing))
            sent += 1
        except Exception as e:
            failed += 1
            print(f"Error analyzing {ticker}: {e}")

    if sent == 0 and failed > 0:
        send_telegram(
            f"⚠️ <b>Signal screener failed</b> — all {failed} ticker(s) errored. "
            f"Check the GitHub Actions run log."
        )
        return 1
    if failed:
        send_telegram(f"⚠️ Signal screener finished with {failed} error(s); see the Actions log.")
    print(f"Done. {sent} signal(s) sent, {failed} failure(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
