import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# STOCKS TO COMPARE
# ----------------------------
stocks = {
    "Apple": "AAPL",
    "Google": "GOOGL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Nvidia": "NVDA"
}

results = []

print(" Analyzing stocks...\n")

for name, ticker in stocks.items():
    try:
        # GET DATA
        raw = yf.download(ticker, period="6mo", interval="1d", progress=False)
        
        # FLATTEN columns (fixes the alignment error)
        raw.columns = raw.columns.get_level_values(0)
        
        df = raw[["Close", "Volume"]].copy()
        df = df.dropna()

        # CALCULATE FEATURES
        df = df.copy()
        df["Return"]     = df["Close"].pct_change()
        df["MA7"]        = df["Close"].rolling(7).mean()
        df["MA30"]       = df["Close"].rolling(30).mean()
        df["Tomorrow"]   = df["Close"].shift(-1)
        df["Will_Go_Up"] = (df["Tomorrow"] > df["Close"]).astype(int)
        df = df.dropna()

        # TRAIN MODEL
        X = df[["Close", "Volume", "Return", "MA7", "MA30"]]
        y = df["Will_Go_Up"]

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)

        # GET CONFIDENCE SCORE
        latest = X.iloc[-1:]
        confidence = model.predict_proba(latest)[0][1] * 100

        # RECENT PERFORMANCE
        last_30_days  = df["Return"].tail(30).mean() * 100
        current_price = float(df["Close"].iloc[-1])

        results.append({
            "Company":               name,
            "Ticker":                ticker,
            "Current Price":         f"${current_price:.2f}",
            "AI Confidence":         confidence,
            "Avg Daily Return (30d)": f"{last_30_days:.3f}%",
            "Recommendation":        "BUY 📈" if confidence > 55 else "WAIT 📉"
        })

        print(f" {name} ({ticker}) analyzed!")

    except Exception as e:
        print(f" Could not analyze {name}: {e}")

# ----------------------------
# SHOW RESULTS
# ----------------------------
print("\n" + "="*60)
print(" AI INVESTMENT PREDICTIONS")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("AI Confidence", ascending=False)

# Format confidence for display
results_df["AI Confidence"] = results_df["AI Confidence"].apply(lambda x: f"{x:.1f}%")

print(results_df.to_string(index=False))

print("\n" + "="*60)
best = results_df.iloc[0]
print(f" BEST INVESTMENT RIGHT NOW: {best['Company']} ({best['Ticker']})")
print(f"   AI Confidence it will go UP: {best['AI Confidence']}")
print(f"   Current Price: {best['Current Price']}")
print(f"   Recommendation: {best['Recommendation']}")
print("="*60)

print("\n  Disclaimer: This is for learning only, not real financial advice!")

results_df.to_csv("investment_predictions.csv", index=False)
print(" Results saved to investment_predictions.csv")