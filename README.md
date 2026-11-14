# Weekly Long-Only Signal Screener

Every Monday shortly after the US market open, GitHub Actions runs this script. It
computes real technical indicators locally, gathers the last 7 days of news, asks
Gemini for a structured BUY/HOLD decision with a rubric-based confidence score,
validates the answer arithmetically, computes a risk-based position size, and
sends the alert to Telegram.

## Setup

1. Push these files to a **private** GitHub repo (root level: `main.py`,
   `requirements.txt`, `.github/workflows/weekly-signals.yml`).
2. Create the secrets (repo Settings > Secrets and variables > Actions > Secrets):
   - `GCP_PROJECT_ID` — your GCP project id (Gemini via Vertex AI; this is the
     required path from Hong Kong and other regions without AI Studio support —
     see https://ai.google.dev/gemini-api/docs/available-regions)
   - `GOOGLE_APPLICATION_CREDENTIALS_JSON` — the **full raw JSON** of a service
     account key (GCP > IAM & Admin > Service Accounts > your account > Keys >
     Add key > JSON). Give the account the **Vertex AI User** role. The script
     writes this to a temp file and uses it as Application Default Credentials.
   - `TELEGRAM_BOT_TOKEN` — from @BotFather
   - `TELEGRAM_CHAT_ID` — message @userinfobot to get your chat id (or send any
     message to your bot and check `getUpdates`)
   - `ACCOUNT_SIZE` (optional) — account value in USD; enables position sizing
3. Optional repo **Variables** (same settings page): `WATCHLIST` (comma-separated,
   e.g. `NVDA,AAPL,MSFT`), `GCP_LOCATION` (default `us-central1`; try
   `asia-east2` for the Hong Kong region if the model is available there),
   `GEMINI_MODEL`, `RISK_PER_TRADE_PCT`, `MAX_POSITION_PCT`.
4. Test it: Actions tab > Weekly Signal Screener > Run workflow.

Local run with Vertex AI: set `GCP_PROJECT_ID` and
`GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` in your `.env` (the JSON
secret variant is for GitHub Actions). The `GEMINI_API_KEY` path still works if
`GCP_PROJECT_ID` is unset and you are in a supported region.

## How decisions are made

- **Indicators computed locally** (never by the LLM): EMA20/50/200, RSI(14),
  MACD histogram, ATR(14), volume vs 20-day average, 52-week range, 1w/1m/3m
  returns. The model only interprets them.
- **Long-only**: the schema physically cannot emit SELL. HOLD means "no edge, do
  not enter".
- **Validated arithmetic**: a BUY must satisfy stop < entry < target,
  reward:risk >= 1.5, entry within 5% of the last close, confidence >= 60. If the
  model gets it wrong, it is retried once with the list of problems; if still
  wrong, the signal is downgraded to HOLD.
- **Position sizing**: with `ACCOUNT_SIZE` set, shares = (account x risk%) /
  (entry - stop), capped at `MAX_POSITION_PCT` of the account. Default risk is
  1% per trade.
- **News freshness**: only headlines from the last 7 days are shown to the model,
  each tagged with its date and publisher. yfinance (>= 1.8.0) is the primary
  source with Google News RSS as automatic fallback.

## Scheduling notes

- Cron is UTC and ignores DST: the run lands at 9:35 AM ET in winter and
  10:35 AM ET in summer. Acceptable for weekly swing signals.
- US market holidays are detected (no bar for today -> skip) so you don't get
  alerts on Memorial Day etc. Set `FORCE_RUN=1` to bypass for manual tests.
- A Friday run (commented out in the workflow) is useful for reviewing exits on
  open positions before the weekend.

## Tuning knobs

| Variable | Default | Meaning |
|---|---|---|
| `WATCHLIST` | NVDA | Comma-separated tickers |
| `GEMINI_MODEL` | gemini-3.6-flash | Any current Gemini model id |
| `ACCOUNT_SIZE` | 0 (off) | USD account value for sizing |
| `RISK_PER_TRADE_PCT` | 1.0 | % of account risked per trade |
| `MAX_POSITION_PCT` | 25.0 | Max % of account in one position |
| `FORCE_RUN` | 0 | 1 = skip holiday check |

In code: `MIN_REWARD_RISK`, `MIN_BUY_CONFIDENCE`, `NEWS_WINDOW_DAYS`,
`MAX_NEWS_ITEMS` near the top of `main.py`.
