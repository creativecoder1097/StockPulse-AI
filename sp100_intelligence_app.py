"""
STOCKPULSE AI - GAMIFIED QUANTITATIVE TERMINAL (v16.0)
---------------------------------------------------------
NEW IN v16.0:
  ✦ Bollinger Bands on candlestick chart
  ✦ MACD indicator panel (with signal line + histogram)
  ✦ Multi-Stock Comparison tab  (overlay normalised price + radar chart)
  ✦ Watchlist  (add/remove tickers, live mini-scorecard per ticker)
  ✦ Trade Journal  (log trades, P&L tracker, equity curve)
  ✦ Earnings & Dividend Calendar  (simulated upcoming events)
  ✦ Sentiment Score  (news-headline tone meter, simulated)
  ✦ CSV Data Export  (download raw OHLCV data)
  ✦ Calmar & Sortino ratios added to Risk Matrix
  ✦ Rolling Sharpe chart  (see how risk-adjusted perf evolves over time)
  ✦ Price-Target Calculator  (entry / target / stop → R:R ratio)
  ✦ Drawdown chart  (visualise underwater periods)
  ✦ Dark/Light theme toggle
  ✦ Keyboard shortcut hints

Theme: Neon-Dark Mode (Pure Black #000000)
Accents: Cyan (#00FFFF), Neon Green (#39FF14)
"""

import os, time, math, logging, warnings, io
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from scipy import stats
from scipy.stats import skew, kurtosis, norm, shapiro
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StockPulse_AI")

VERSION       = "16.0.0-STOCKPULSE-NEON"
PLATFORM_NAME = "StockPulse AI"
DEFAULT_TICKER = "NVDA"
TRADING_DAYS   = 252

# ─────────────────────────────────────────────────────────────
# 1. QUANTITATIVE BACKEND
# ─────────────────────────────────────────────────────────────

class DataEngine:
    @staticmethod
    @st.cache_data(ttl=1800)
    def fetch_validated_payload(ticker, api_key=None):
        try:
            if not api_key:
                df     = DataEngine.generate_advanced_synthetic(ticker)
                status = "SIMULATION_NODE"
            else:
                end_ts   = int(time.time())
                start_ts = end_ts - (5 * 365 * 24 * 60 * 60)
                url = (f"https://finnhub.io/api/v1/stock/candle"
                       f"?symbol={ticker}&resolution=D"
                       f"&from={start_ts}&to={end_ts}&token={api_key}")
                res = requests.get(url, timeout=15).json()
                if res.get("s") == "ok":
                    df = pd.DataFrame(
                        {"Open": res["o"], "High": res["h"],
                         "Low": res["l"], "Close": res["c"],
                         "Volume": res["v"]},
                        index=pd.to_datetime(res["t"], unit="s"),
                    )
                    status = "CORE_LIVE_LINK"
                else:
                    df     = DataEngine.generate_advanced_synthetic(ticker)
                    status = "FALLBACK_ACTIVE"
            df = DataEngine.sanitize(df)
            return df, status
        except Exception as e:
            logger.error(e)
            return None, "NODE_FAILURE"

    @staticmethod
    def sanitize(df):
        if df is None or df.empty:
            return None
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        log_rets = np.log((df["Close"] + 1e-9) / (df["Close"].shift(1) + 1e-9)).fillna(0)
        z_scores = np.abs(stats.zscore(log_rets))
        df = df[z_scores < 4]
        df = df.resample("B").ffill().bfill()
        if "Open" not in df.columns:
            df["Open"] = df["Close"].shift(1).fillna(df["Close"])
        return df

    @staticmethod
    def generate_advanced_synthetic(ticker):
        np.random.seed(abs(hash(ticker)) % (10 ** 8))
        periods = 1260
        dates   = pd.date_range(end=datetime.today(), periods=periods, freq="B")
        s0, mu, v0 = 150.0, 0.0005, 0.02
        vol = [v0]
        for _ in range(1, periods):
            vol.append(max(0.01, vol[-1] + np.random.normal(0, 0.002)))
        rets  = np.random.normal(mu, vol, periods)
        price = s0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "Close":  price,
            "High":   price * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            "Low":    price * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            "Volume": np.random.randint(5_000_000, 40_000_000, periods),
        }, index=dates)
        df["Open"] = df["Close"].shift(1).fillna(df["Close"] * 0.99)
        return df


class QuantCore:
    @staticmethod
    def solve_metrics(df):
        try:
            rets = np.log(df["Close"] / df["Close"].shift(1)).dropna()
            rets = rets.replace([np.inf, -np.inf], 0).fillna(0)
            _, shapiro_p  = shapiro(rets[-200:])
            half = len(rets) // 2
            levene_p = stats.levene(rets[:half], rets[half:]).pvalue
            bench  = np.random.normal(rets.mean(), rets.std() * 0.95, len(rets))
            X = np.vstack([np.ones(len(bench)), bench]).T
            coeffs = np.linalg.lstsq(X, rets.values, rcond=None)[0]
            max_p, min_p = df["High"].max(), df["Low"].min()
            rng = max_p - min_p
            fibs = {"23.6%": max_p - 0.236*rng, "38.2%": max_p - 0.382*rng, "61.8%": max_p - 0.618*rng}
            ann_ret = rets.mean() * TRADING_DAYS
            ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
            sharpe  = ann_ret / ann_vol if ann_vol != 0 else 0

            # Sortino
            neg_rets = rets[rets < 0]
            downside_dev = neg_rets.std() * np.sqrt(TRADING_DAYS) if len(neg_rets) > 0 else 1e-9
            sortino = ann_ret / downside_dev if downside_dev != 0 else 0

            # Max Drawdown + Calmar
            cum = (1 + rets).cumprod()
            roll_max = cum.cummax()
            drawdown = (cum - roll_max) / roll_max
            max_dd = drawdown.min()
            calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

            var_95  = np.percentile(rets, 5)
            cvar_95 = rets[rets <= var_95].mean()
            sk  = float(skew(rets))
            ku  = float(kurtosis(rets))

            # Rolling Sharpe (63-day windows)
            roll_sharpe = (rets.rolling(63).mean() * TRADING_DAYS) / (rets.rolling(63).std() * np.sqrt(TRADING_DAYS))

            # Drawdown series
            drawdown_series = drawdown

            return {
                "rets": rets, "alpha": coeffs[0] * TRADING_DAYS,
                "beta": coeffs[1], "sharpe": sharpe, "sortino": sortino,
                "calmar": calmar, "ann_ret": ann_ret, "ann_vol": ann_vol,
                "max_dd": max_dd, "var_95": var_95, "cvar_95": cvar_95,
                "skew": sk, "kurt": ku,
                "is_normal": shapiro_p > 0.05,
                "is_stationary": levene_p > 0.05,
                "fibs": fibs, "corr": float(np.corrcoef(rets, bench)[0, 1]),
                "roll_sharpe": roll_sharpe, "drawdown_series": drawdown_series,
            }
        except:
            return None

    @staticmethod
    def monte_carlo(df, sims=300, horizon=30):
        rets  = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        mu, sigma = rets.mean(), rets.std()
        s0 = df["Close"].iloc[-1]
        paths = np.zeros((horizon, sims))
        for i in range(sims):
            r = np.random.normal(mu, sigma, horizon)
            paths[:, i] = s0 * np.exp(np.cumsum(r))
        return paths

    @staticmethod
    def compute_macd(df, fast=12, slow=26, signal=9):
        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram   = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def compute_bollinger(df, window=20, num_std=2):
        ma  = df["Close"].rolling(window).mean()
        std = df["Close"].rolling(window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        return ma, upper, lower

    @staticmethod
    def sentiment_score(ticker):
        """Deterministic simulated sentiment based on ticker hash."""
        np.random.seed(abs(hash(ticker + "sent")) % (10**6))
        score = float(np.random.uniform(20, 90))
        headlines = [
            (f"{ticker} beats earnings estimate", np.random.choice(["Bullish","Neutral","Bullish"])),
            (f"Analyst upgrades {ticker} price target", np.random.choice(["Bullish","Bullish","Neutral"])),
            (f"Sector rotation away from {ticker} peers", np.random.choice(["Bearish","Neutral"])),
            (f"{ticker} announces share buyback programme", np.random.choice(["Bullish","Neutral"])),
            (f"Macro headwinds weigh on {ticker}", np.random.choice(["Bearish","Neutral","Bearish"])),
        ]
        return score, headlines

    @staticmethod
    def earnings_calendar(ticker):
        """Generate simulated upcoming events."""
        np.random.seed(abs(hash(ticker + "cal")) % (10**6))
        today = datetime.today()
        events = []
        for i in range(1, 5):
            delta = np.random.randint(5, 90 * i)
            ev_date = today + timedelta(days=int(delta))
            kind = np.random.choice(["Earnings Release", "Dividend Ex-Date", "Analyst Day", "Product Event"])
            est_eps = round(np.random.uniform(0.5, 5.0), 2)
            events.append({"Date": ev_date.strftime("%d %b %Y"), "Event": kind,
                           "Est. EPS": f"${est_eps}" if "Earnings" in kind else "—",
                           "Days Away": int(delta)})
        return sorted(events, key=lambda x: x["Days Away"])


class AIModel:
    @staticmethod
    def predict_trajectory(df):
        try:
            d = df.copy()
            d["MA20"] = d["Close"].rolling(20).mean()
            d["MA50"] = d["Close"].rolling(50).mean()
            diff = d["Close"].diff()
            rs = (diff.clip(lower=0).rolling(14).mean() /
                  diff.clip(upper=0).abs().rolling(14).mean().replace(0, 1e-9))
            d["RSI"]    = 100 - (100 / (1 + rs))
            d["Vol20"]  = d["Close"].rolling(20).std()
            d["Target"] = d["Close"].shift(-5)
            feats = ["MA20", "MA50", "RSI", "Vol20"]
            d = d.dropna()
            if len(d) < 100:
                return None, None
            X, y = d[feats], d["Target"]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
            model.fit(X_s, y)
            last_feat = X.iloc[-1].values.reshape(1, -1)
            path = [df["Close"].iloc[-1]]
            for _ in range(7):
                s = scaler.transform(last_feat)
                p = model.predict(s)[0]
                path.append(p)
                last_feat = last_feat * (1 + np.random.normal(0, 0.002, last_feat.shape))
            vol  = df["Close"].pct_change().std()
            conf = [vol * p * (i + 1) ** 0.5 for i, p in enumerate(path[1:])]
            return path[1:], conf
        except:
            return None, None


# ─────────────────────────────────────────────────────────────
# 2. HELPER UTILITIES
# ─────────────────────────────────────────────────────────────

def risk_label(sharpe, max_dd, ann_vol):
    score = 0
    if sharpe > 1.5: score += 2
    elif sharpe > 0.5: score += 1
    if max_dd > -0.30: score += 0
    elif max_dd > -0.15: score += 1
    else: score += 2
    if ann_vol < 0.20: score += 2
    elif ann_vol < 0.35: score += 1
    if score >= 5: return "LOW RISK",    "#39FF14", "🟢"
    if score >= 3: return "MEDIUM RISK", "#FFD700", "🟡"
    return              "HIGH RISK",   "#FF3131", "🔴"

def pct_delta(df):
    if len(df) < 2: return 0, 0, 0
    d1  = float(df["Close"].pct_change().dropna().iloc[-1]) * 100
    d5  = float((df["Close"].iloc[-1] / df["Close"].iloc[-6]  - 1) * 100) if len(df) >= 6  else 0
    d30 = float((df["Close"].iloc[-1] / df["Close"].iloc[-31] - 1) * 100) if len(df) >= 31 else 0
    return d1, d5, d30

GLOSSARY = {
    "Alpha":        "Annual excess return vs the benchmark. Positive alpha = outperforming the market.",
    "Beta":         "Sensitivity to market moves. Beta > 1 means more volatile than the market; < 1 means less.",
    "Sharpe":       "Risk-adjusted return. Above 1.0 is good; above 2.0 is excellent.",
    "Sortino":      "Like Sharpe, but only penalises downside volatility. Higher is better.",
    "Calmar":       "Annual return divided by max drawdown. Higher = better recovery per unit of risk.",
    "Max Drawdown": "Largest peak-to-trough loss. Smaller (less negative) is better.",
    "VaR 95%":      "Value at Risk: on the worst 5% of days, losses exceed this number.",
    "CVaR 95%":     "Conditional VaR: average loss on those worst 5% of days.",
    "Skewness":     "Return asymmetry. Negative skew = more frequent downside surprises.",
    "Kurtosis":     "Fat-tail measure. High kurtosis = extreme moves happen more than normal.",
    "Fibonacci":    "Support/resistance levels traders watch. Price tends to react near these.",
    "RSI":          "Relative Strength Index. Above 70 = overbought, below 30 = oversold.",
    "MACD":         "Moving Average Convergence/Divergence. Momentum indicator showing trend strength and direction.",
    "Bollinger Bands": "Volatility envelope around a moving average. Wide bands = high volatility.",
    "Monte Carlo":  "Hundreds of simulated price paths based on historical volatility.",
    "R:R Ratio":    "Risk-to-Reward. A 1:2 ratio means risking $1 to potentially gain $2.",
    "Rolling Sharpe": "Sharpe ratio calculated over a rolling window — shows how risk-adj. performance changes over time.",
}


# ─────────────────────────────────────────────────────────────
# 3. STYLES
# ─────────────────────────────────────────────────────────────

def apply_neon_styles(dark=True):
    if dark:
        bg, fg, card_bg, border_col = "#000000", "#FFFFFF", "rgba(8,8,15,0.95)", "rgba(0,255,255,0.25)"
        sub_fg = "#94A3B8"
    else:
        bg, fg, card_bg, border_col = "#F0F4FF", "#0A0A1A", "rgba(255,255,255,0.97)", "rgba(0,100,200,0.25)"
        sub_fg = "#4A5568"

    st.markdown(f"""
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
        .stApp {{ background-color: {bg} !important; color: {fg} !important; font-family: 'Syne', sans-serif !important; }}
        .bento-card {{
            background: {card_bg};
            border: 1px solid {border_col};
            border-radius: 20px; padding: 24px 28px; margin-bottom: 18px;
            box-shadow: 0 4px 40px rgba(0,255,255,0.07);
            transition: border 0.3s, box-shadow 0.3s, transform 0.25s;
            backdrop-filter: blur(12px);
        }}
        .bento-card:hover {{ border-color: #39FF14; box-shadow: 0 0 28px rgba(57,255,20,0.18); transform: translateY(-2px) scale(1.005); }}
        .hero-headline {{
            font-family: 'Orbitron', sans-serif; font-size: 4.2rem; font-weight: 900;
            line-height: 1; letter-spacing: -1px;
            background: linear-gradient(135deg, #00FFFF 0%, #39FF14 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .hero-sub {{ font-family: 'Syne', sans-serif; font-size: 0.85rem; letter-spacing: 5px; color: #00FFFF; font-weight: 700; text-transform: uppercase; }}
        .qstat {{
            display: inline-block; background: rgba(0,255,255,0.06);
            border: 1px solid rgba(0,255,255,0.18); border-radius: 12px;
            padding: 10px 18px; margin: 4px;
            font-family: 'Orbitron', sans-serif; font-size: 0.72rem; color: {sub_fg}; text-align: center;
        }}
        .qstat .val {{ font-size: 1.15rem; font-weight: 700; color: {fg}; display: block; }}
        .qstat .pos {{ color: #39FF14 !important; }}
        .qstat .neg {{ color: #FF3131 !important; }}
        .section-label {{ font-family: 'Orbitron', sans-serif; font-size: 0.65rem; letter-spacing: 4px; color: #475569; text-transform: uppercase; margin-bottom: 6px; }}
        .explain-box {{ background: rgba(0,255,255,0.04); border-left: 3px solid #00FFFF; border-radius: 0 10px 10px 0; padding: 10px 14px; font-size: 0.82rem; color: {sub_fg}; margin-top: 8px; line-height: 1.55; }}
        .badge-live  {{ color:#39FF14; background:rgba(57,255,20,0.12);  border:1px solid #39FF14;  border-radius:20px; padding:3px 12px; font-size:0.7rem; font-family:Orbitron; }}
        .badge-sim   {{ color:#FFD700; background:rgba(255,215,0,0.10);  border:1px solid #FFD700; border-radius:20px; padding:3px 12px; font-size:0.7rem; font-family:Orbitron; }}
        .badge-err   {{ color:#FF3131; background:rgba(255,49,49,0.10);  border:1px solid #FF3131;  border-radius:20px; padding:3px 12px; font-size:0.7rem; font-family:Orbitron; }}
        .step-circle {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 26px; height: 26px; background: linear-gradient(135deg,#00FFFF,#39FF14);
            border-radius: 50%; font-weight: 900; font-size: 0.75rem; color: #000; margin-right: 8px; flex-shrink: 0;
        }}
        .step-row {{ display:flex; align-items:center; margin-bottom:12px; font-size:0.88rem; color:#CBD5E1; }}
        .stButton button {{
            background: linear-gradient(135deg, #00FFFF, #2563EB) !important;
            color: #000 !important; font-family: 'Orbitron', sans-serif !important;
            font-weight: 700 !important; border-radius: 50px !important; border: none !important;
            padding: 10px 30px !important; letter-spacing: 1px;
            box-shadow: 0 0 12px rgba(0,255,255,0.35); transition: all 0.25s;
        }}
        .stButton button:hover {{ box-shadow: 0 0 28px rgba(0,255,255,0.7); transform: translateY(-2px); }}
        [data-testid="stSidebar"] {{ display: none !important; }}
        section[data-testid="stSidebarNav"] {{ display: none !important; }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px; background: rgba(0,0,0,0.6) !important;
            border-bottom: 1px solid rgba(0,255,255,0.15) !important;
            padding: 0 8px; border-radius: 16px 16px 0 0; flex-wrap: wrap;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: #475569 !important; font-family: 'Orbitron', sans-serif !important;
            font-size: 0.65rem !important; letter-spacing: 1px !important;
            padding: 10px 12px !important; border-radius: 10px 10px 0 0 !important;
        }}
        .stTabs [aria-selected="true"] {{ color: #00FFFF !important; border-bottom: 2px solid #00FFFF !important; background: rgba(0,255,255,0.06) !important; }}
        [data-testid="stMetric"] label {{ color:#475569 !important; font-size:0.72rem !important; font-family:Orbitron !important; letter-spacing:2px; }}
        [data-testid="stMetricValue"]  {{ color:{fg} !important; font-family:Orbitron !important; }}
        .settings-card {{ background: rgba(0,8,20,0.97); border: 1px solid rgba(0,255,255,0.2); border-radius: 16px; padding: 24px 28px; margin-bottom: 16px; }}
        .sentiment-bar {{ height: 12px; border-radius: 6px; background: linear-gradient(90deg, #FF3131, #FFD700, #39FF14); position: relative; }}
        .watch-row {{ display:flex; align-items:center; justify-content:space-between; padding:10px 0; border-bottom:1px solid #1E293B; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        .stDivider {{ border-color: #1E293B !important; }}
        hr {{ border-color: #1E293B !important; }}
        </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 4. COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────

def section(label):
    st.markdown(f"<p class='section-label'>{label}</p>", unsafe_allow_html=True)

def explain(text):
    st.markdown(f"<div class='explain-box'>💡 {text}</div>", unsafe_allow_html=True)

def status_badge(status):
    if status == "CORE_LIVE_LINK":     return "<span class='badge-live'>● LIVE DATA</span>"
    if status in ("SIMULATION_NODE", "FALLBACK_ACTIVE"): return "<span class='badge-sim'>◑ SIMULATED</span>"
    return "<span class='badge-err'>✕ ERROR</span>"

def color_val(v):
    return "pos" if v >= 0 else "neg"

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=20, b=0),
    font=dict(family="Syne", color="#94A3B8"),
)

def quick_stats_bar(df, ticker):
    d1, d5, d30 = pct_delta(df)
    high52  = df["Close"].tail(252).max()
    low52   = df["Close"].tail(252).min()
    avg_vol = int(df["Volume"].tail(20).mean())
    lp      = df["Close"].iloc[-1]
    st.markdown(f"""
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 18px 0;">
            <div class="qstat"><span style="font-size:0.65rem;">ASSET</span><span class="val">{ticker}</span></div>
            <div class="qstat"><span style="font-size:0.65rem;">SPOT PRICE</span><span class="val">${lp:,.2f}</span></div>
            <div class="qstat"><span style="font-size:0.65rem;">1-DAY</span><span class="val {color_val(d1)}">{d1:+.2f}%</span></div>
            <div class="qstat"><span style="font-size:0.65rem;">5-DAY</span><span class="val {color_val(d5)}">{d5:+.2f}%</span></div>
            <div class="qstat"><span style="font-size:0.65rem;">30-DAY</span><span class="val {color_val(d30)}">{d30:+.2f}%</span></div>
            <div class="qstat"><span style="font-size:0.65rem;">52W HIGH</span><span class="val">${high52:,.2f}</span></div>
            <div class="qstat"><span style="font-size:0.65rem;">52W LOW</span><span class="val">${low52:,.2f}</span></div>
            <div class="qstat"><span style="font-size:0.65rem;">AVG VOL 20D</span><span class="val">{avg_vol:,}</span></div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 5. CHART BUILDERS
# ─────────────────────────────────────────────────────────────

def chart_candlestick(raw, ai_preds, conf, window=120, show_bb=True):
    section("AI PULSE — TREND DISCOVERY")
    ma_bb, bb_upper, bb_lower = QuantCore.compute_bollinger(raw)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.78, 0.22])
    tail = raw.tail(window)
    fig.add_trace(go.Candlestick(
        x=tail.index, open=tail["Open"], high=tail["High"],
        low=tail["Low"], close=tail["Close"], name="Price",
        increasing_line_color="#39FF14", decreasing_line_color="#FF3131",
    ), row=1, col=1)
    ma20 = raw["Close"].rolling(20).mean().tail(window)
    ma50 = raw["Close"].rolling(50).mean().tail(window)
    fig.add_trace(go.Scatter(x=tail.index, y=ma20, name="MA 20",
                             line=dict(color="#00FFFF", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=tail.index, y=ma50, name="MA 50",
                             line=dict(color="#FFD700", width=1.2, dash="dot")), row=1, col=1)
    if show_bb:
        bb_u = bb_upper.tail(window)
        bb_l = bb_lower.tail(window)
        fig.add_trace(go.Scatter(x=tail.index, y=bb_u, name="BB Upper",
                                 line=dict(color="rgba(255,100,255,0.5)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=tail.index, y=bb_l, name="BB Lower",
                                 fill="tonexty", fillcolor="rgba(255,100,255,0.05)",
                                 line=dict(color="rgba(255,100,255,0.5)", width=1)), row=1, col=1)
    if ai_preds:
        f_dates = [raw.index[-1] + timedelta(days=i) for i in range(1, 8)]
        upper   = [p + c for p, c in zip(ai_preds, conf)]
        lower   = [p - c for p, c in zip(ai_preds, conf)]
        fig.add_trace(go.Scatter(x=f_dates, y=upper, fill=None, mode="lines",
                                 line=dict(color="rgba(0,255,255,0.0)"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=f_dates, y=lower, fill="tonexty",
                                 fillcolor="rgba(0,255,255,0.08)", mode="lines",
                                 line=dict(color="rgba(0,255,255,0.0)"),
                                 name="Confidence Band"), row=1, col=1)
        fig.add_trace(go.Scatter(x=f_dates, y=ai_preds, name="7-Day AI Forecast",
                                 line=dict(color="#00FFFF", width=2.5, dash="dash"),
                                 mode="lines+markers",
                                 marker=dict(size=6, color="#00FFFF")), row=1, col=1)
    fig.add_trace(go.Bar(x=tail.index, y=tail["Volume"], name="Volume",
                         marker_color="#1E293B"), row=2, col=1)
    fig.update_layout(**CHART_LAYOUT, height=560, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    explain("Green candles = up day | Red = down day. Purple band = Bollinger Bands (volatility envelope). "
            "Cyan dashed line = AI 7-day forecast with confidence shading.")


def chart_macd(df):
    section("MACD — MOMENTUM & TREND STRENGTH")
    macd_line, signal_line, histogram = QuantCore.compute_macd(df)
    tail_n = 120
    idx = df.index[-tail_n:]
    m = macd_line.tail(tail_n)
    s = signal_line.tail(tail_n)
    h = histogram.tail(tail_n)
    colors = ["#39FF14" if v >= 0 else "#FF3131" for v in h]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=h, name="Histogram", marker_color=colors, opacity=0.6))
    fig.add_trace(go.Scatter(x=idx, y=m, name="MACD", line=dict(color="#00FFFF", width=1.8)))
    fig.add_trace(go.Scatter(x=idx, y=s, name="Signal", line=dict(color="#FFD700", width=1.5, dash="dot")))
    fig.add_hline(y=0, line_color="#475569", line_dash="dot")
    fig.update_layout(**CHART_LAYOUT, height=260,
                      legend=dict(orientation="h", font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    last_hist = float(histogram.iloc[-1])
    cross     = float(macd_line.iloc[-1]) - float(signal_line.iloc[-1])
    if cross > 0:
        explain(f"MACD is above its signal line (histogram = {last_hist:+.4f}) — bullish momentum building. "
                "A widening gap suggests strengthening uptrend.")
    else:
        explain(f"MACD is below its signal line (histogram = {last_hist:+.4f}) — bearish momentum. "
                "Watch for a cross above the signal line for a potential reversal.")


def chart_rsi(df):
    section("RSI — MOMENTUM INDICATOR")
    diff = df["Close"].diff()
    rs   = (diff.clip(lower=0).rolling(14).mean() /
            diff.clip(upper=0).abs().rolling(14).mean().replace(0, 1e-9))
    rsi  = (100 - (100 / (1 + rs))).tail(120)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, fill="tozeroy",
                             fillcolor="rgba(0,255,255,0.08)",
                             line=dict(color="#00FFFF", width=1.8), name="RSI"))
    fig.add_hline(y=70, line_dash="dash", line_color="#FF3131",
                  annotation_text="Overbought (70)", annotation_font_color="#FF3131", annotation_font_size=10)
    fig.add_hline(y=30, line_dash="dash", line_color="#39FF14",
                  annotation_text="Oversold (30)",   annotation_font_color="#39FF14", annotation_font_size=10)
    fig.update_layout(**CHART_LAYOUT, height=240, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    lr = float(rsi.iloc[-1])
    if lr > 70:   explain(f"⚠️ RSI = {lr:.1f} — **Overbought**. Prices could pull back.")
    elif lr < 30: explain(f"✅ RSI = {lr:.1f} — **Oversold**. Potential buying opportunity.")
    else:         explain(f"➡️ RSI = {lr:.1f} — Momentum is **neutral**.")


def chart_fibonacci(df, fibs):
    section("FIBONACCI RETRACEMENT LEVELS")
    tail = df.tail(180)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=tail.index, y=tail["Close"],
                             line=dict(color="#FFFFFF", width=1.5), name="Price"))
    colors = {"23.6%": "#FFD700", "38.2%": "#00FFFF", "61.8%": "#39FF14"}
    for label, level in fibs.items():
        fig.add_hline(y=level, line_dash="dot", line_color=colors[label],
                      annotation_text=f"{label} — ${level:,.2f}",
                      annotation_font_color=colors[label], annotation_font_size=10)
    fig.update_layout(**CHART_LAYOUT, height=280)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    explain("38.2% is often the first major support/resistance. 61.8% is the 'golden ratio' level.")


def chart_return_dist(rets):
    section("RETURN DISTRIBUTION")
    mu, sigma = rets.mean(), rets.std()
    x_range   = np.linspace(rets.min(), rets.max(), 200)
    normal_pdf = norm.pdf(x_range, mu, sigma)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets, nbinsx=60, name="Actual Returns",
                               marker_color="#00FFFF", opacity=0.55, histnorm="probability density"))
    fig.add_trace(go.Scatter(x=x_range, y=normal_pdf, name="Normal Curve",
                             line=dict(color="#39FF14", width=2)))
    var = np.percentile(rets, 5)
    fig.add_vline(x=var, line_dash="dash", line_color="#FF3131",
                  annotation_text=f" VaR 95% = {var*100:.2f}%",
                  annotation_font_color="#FF3131", annotation_font_size=11)
    fig.update_layout(**CHART_LAYOUT, height=280, showlegend=True, legend=dict(font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    explain("Histogram of daily returns. Red dashed line = VaR 95% — on the worst 5% of days losses exceeded this.")


def chart_monte_carlo(paths, last_price):
    section("MONTE CARLO SIMULATION — 30-DAY OUTLOOK")
    fig = go.Figure()
    for i in range(min(200, paths.shape[1])):
        fig.add_trace(go.Scatter(y=paths[:, i], mode="lines",
                                 line=dict(color="rgba(0,255,255,0.05)", width=1), showlegend=False))
    p10 = np.percentile(paths, 10, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p90 = np.percentile(paths, 90, axis=1)
    fig.add_trace(go.Scatter(y=p10, name="10th %ile", line=dict(color="#FF3131", width=2)))
    fig.add_trace(go.Scatter(y=p50, name="Median",    line=dict(color="#FFD700", width=2.5)))
    fig.add_trace(go.Scatter(y=p90, name="90th %ile", line=dict(color="#39FF14", width=2)))
    fig.add_hline(y=last_price, line_dash="dot", line_color="#FFFFFF",
                  annotation_text="Current Price", annotation_font_color="#FFFFFF", annotation_font_size=10)
    fig.update_layout(**CHART_LAYOUT, height=340,
                      xaxis_title="Trading Days", yaxis_title="Simulated Price ($)",
                      legend=dict(font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    final = paths[-1, :]
    bull  = float(np.mean(final > last_price) * 100)
    med_f = float(np.median(final))
    explain(f"{paths.shape[1]} simulated futures. Median 30-day price: ${med_f:,.2f}. "
            f"Probability above current price: {bull:.0f}% | Below: {100-bull:.0f}%.")


def chart_rolling_sharpe(roll_sharpe):
    section("ROLLING SHARPE RATIO (63-DAY WINDOW)")
    rs = roll_sharpe.dropna().tail(252)
    colors = ["#39FF14" if v > 1 else "#FFD700" if v > 0 else "#FF3131" for v in rs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs, mode="lines",
                             line=dict(color="#00FFFF", width=1.5), name="Rolling Sharpe",
                             fill="tozeroy", fillcolor="rgba(0,255,255,0.06)"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#39FF14",
                  annotation_text="Good (1.0)", annotation_font_color="#39FF14", annotation_font_size=10)
    fig.add_hline(y=0.0, line_dash="dot", line_color="#FF3131")
    fig.update_layout(**CHART_LAYOUT, height=240, yaxis_title="Sharpe")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    explain("A rolling Sharpe above 1.0 (green zone) means risk-adjusted performance is solid for that period. "
            "Dips below 0 indicate periods where losses outweighed the risk taken.")


def chart_drawdown(drawdown_series):
    section("DRAWDOWN CHART — UNDERWATER PERIODS")
    dd = drawdown_series.tail(504)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, mode="lines",
                             fill="tozeroy", fillcolor="rgba(255,49,49,0.12)",
                             line=dict(color="#FF3131", width=1.5), name="Drawdown %"))
    fig.update_layout(**CHART_LAYOUT, height=220,
                      yaxis_title="Drawdown (%)", yaxis_ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    explain("Shows how far the portfolio was below its previous peak at each point in time. "
            "Deeper / longer red zones = more painful bear markets.")


def chart_comparison(dfs_dict):
    """Normalised price comparison for multiple tickers."""
    section("NORMALISED PRICE COMPARISON (Base = 100)")
    fig = go.Figure()
    palette = ["#00FFFF", "#39FF14", "#FFD700", "#FF3131", "#BF5FFF"]
    for i, (tkr, df) in enumerate(dfs_dict.items()):
        norm_price = df["Close"] / df["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(x=df.index, y=norm_price,
                                 mode="lines", name=tkr,
                                 line=dict(color=palette[i % len(palette)], width=2)))
    fig.update_layout(**CHART_LAYOUT, height=380, yaxis_title="Normalised Price",
                      legend=dict(orientation="h", font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_radar(metrics_dict):
    """Radar/spider chart comparing multiple tickers across key metrics."""
    section("MULTI-STOCK RADAR — RISK & RETURN PROFILE")
    categories = ["Sharpe", "Ann Return", "Low Volatility", "Low Drawdown", "Alpha"]
    palette = ["#00FFFF", "#39FF14", "#FFD700", "#FF3131", "#BF5FFF"]
    fig = go.Figure()
    for i, (tkr, m) in enumerate(metrics_dict.items()):
        values = [
            min(max(m["sharpe"] / 3, 0), 1),
            min(max(m["ann_ret"] / 0.5, 0), 1),
            1 - min(m["ann_vol"] / 0.8, 1),
            1 - min(abs(m["max_dd"]) / 0.6, 1),
            min(max(m["alpha"] / 0.3 + 0.5, 0), 1),
        ]
        values += [values[0]]  # close the loop
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", fillcolor=f"rgba{tuple(int(palette[i%5].lstrip('#')[j:j+2],16) for j in (0,2,4)) + (0.1,)}",
            line=dict(color=palette[i % len(palette)], width=2),
            name=tkr,
        ))
    fig.update_layout(**CHART_LAYOUT, height=380,
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                                  gridcolor="#1E293B", linecolor="#1E293B"),
                                 angularaxis=dict(gridcolor="#1E293B")),
                      legend=dict(font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────
# 6. SETTINGS TAB
# ─────────────────────────────────────────────────────────────

def render_settings_tab():
    st.markdown("""
        <div style='background:rgba(0,255,255,0.04); border-radius:14px;
                    padding:18px 22px; margin-bottom:20px; border:1px solid rgba(0,255,255,0.12);'>
            <p style='color:#00FFFF; font-family:Orbitron; font-size:0.75rem; letter-spacing:3px; margin-bottom:12px;'>⚡ QUICK-START GUIDE</p>
            <div class='step-row'><span class='step-circle'>1</span>Enter a stock ticker below</div>
            <div class='step-row'><span class='step-circle'>2</span>Optionally paste your Finnhub API key for live data</div>
            <div class='step-row'><span class='step-circle'>3</span>Click <b>Run Analysis</b>, then explore any tab</div>
        </div>""", unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
        section("ASSET SELECTION")
        ticker = st.text_input("📌 Stock Ticker",
                               st.session_state.get("ticker", DEFAULT_TICKER),
                               key="settings_ticker",
                               help="Any US stock symbol, e.g. AAPL, TSLA, MSFT.").upper()
        api_key = st.text_input("🔑 Finnhub API Key (optional)", type="password",
                                key="settings_api_key",
                                help="Leave blank for simulated data. Get a free key at finnhub.io.")
        st.caption("🔒 Keys are never stored or sent anywhere except Finnhub.")
        st.markdown("</div>", unsafe_allow_html=True)

    with sc2:
        st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
        section("CHART & SIMULATION CONTROLS")
        window = st.slider("📊 Chart Window (days)", 30, 252,
                           st.session_state.get("window", 120), 10,
                           key="settings_window")
        mc_sims = st.slider("🎲 Monte Carlo Simulations", 100, 1000,
                            st.session_state.get("mc_sims", 300), 100,
                            key="settings_mc_sims")
        show_bb = st.toggle("📐 Show Bollinger Bands", value=st.session_state.get("show_bb", True),
                            key="settings_show_bb")
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", True),
                              key="settings_dark_mode")
        st.markdown("</div>", unsafe_allow_html=True)

    col_run, col_reset, _ = st.columns([1, 1, 2])
    with col_run:
        run_btn = st.button("▶ Run Analysis", use_container_width=True)
    with col_reset:
        reset_btn = st.button("🔄 Reset Cache", use_container_width=True)
    if reset_btn:
        st.cache_data.clear()
        st.rerun()

    st.session_state["ticker"]    = ticker
    st.session_state["api_key"]   = api_key
    st.session_state["window"]    = window
    st.session_state["mc_sims"]   = mc_sims
    st.session_state["show_bb"]   = show_bb
    st.session_state["dark_mode"] = dark_mode

    if run_btn:
        st.session_state["run_triggered"] = True
        st.rerun()

    st.divider()
    st.markdown("<p style='color:#475569; font-size:0.72rem; font-family:Orbitron;'>STOCKPULSE AI · v16.0 · FOR EDUCATIONAL USE ONLY</p>",
                unsafe_allow_html=True)
    return ticker, api_key, window, mc_sims, show_bb


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title=PLATFORM_NAME, page_icon="⚡",
                       layout="wide", initial_sidebar_state="collapsed")

    dark_mode = st.session_state.get("dark_mode", True)
    apply_neon_styles(dark_mode)

    # ── Hero ──────────────────────────────────────────────────
    _, col_hero, _ = st.columns([1, 2, 1])
    with col_hero:
        st.markdown("<h1 class='hero-headline' style='text-align:center;'>STOCKPULSE AI</h1>",
                    unsafe_allow_html=True)
        st.markdown("<p class='hero-sub' style='text-align:center;'>Gamified Market Intelligence Terminal · v16</p>",
                    unsafe_allow_html=True)

    # ── Onboarding ────────────────────────────────────────────
    with st.expander("👋 New here? Click to learn how to use StockPulse AI", expanded=False):
        oa, ob, oc = st.columns(3)
        with oa:
            st.markdown("""**📡 Data Sources**
- No API key? App runs on *simulated* data — great for learning.
- Have a Finnhub key? Enter it in ⚙ Settings for real prices.
- Free key at [finnhub.io](https://finnhub.io).""")
        with ob:
            st.markdown("""**📊 What's New in v16**
- Bollinger Bands & MACD charts
- Multi-Stock Comparison & Radar
- Watchlist + Trade Journal
- Earnings Calendar & Sentiment
- CSV Export, Drawdown Chart
- Rolling Sharpe, Sortino, Calmar""")
        with oc:
            st.markdown("""**🎮 Navigation Tips**
- Start at ⚙ Settings → enter ticker → Run Analysis
- 🏠 Dashboard = price overview
- 📈 Technicals = RSI, MACD, Bollinger, Fibonacci
- 📊 Compare = multi-stock overlay
- 📋 Journal = log your trades""")

    # ── TOP TABS ──────────────────────────────────────────────
    tabs = st.tabs([
        "⚙ SETTINGS",
        "🏠 DASHBOARD",
        "📈 TECHNICALS",
        "🎲 MONTE CARLO",
        "🧪 HYPOTHESIS LAB",
        "📐 RISK MATRIX",
        "📊 COMPARE",
        "👁 WATCHLIST",
        "📅 CALENDAR",
        "📰 SENTIMENT",
        "💼 PORTFOLIO",
        "📋 JOURNAL",
        "🌐 COMMUNITY",
        "📖 GLOSSARY",
    ])

    # ── TAB 0: SETTINGS ──────────────────────────────────────
    with tabs[0]:
        ticker, api_key, window, mc_sims, show_bb = render_settings_tab()

    # Read from session state
    ticker  = st.session_state.get("ticker",  DEFAULT_TICKER)
    api_key = st.session_state.get("api_key", "")
    window  = st.session_state.get("window",  120)
    mc_sims = st.session_state.get("mc_sims", 300)
    show_bb = st.session_state.get("show_bb", True)

    # ── Load & Compute ────────────────────────────────────────
    with st.spinner(f"Synchronising asset node for **{ticker}**…"):
        raw, status = DataEngine.fetch_validated_payload(ticker, api_key)

    if raw is None or len(raw) < 60:
        for tab in tabs[1:]:
            with tab:
                st.error("⚡ NODE DISCONNECT — Could not load data. Go to ⚙ Settings and verify the ticker.")
        return

    with st.spinner("Running quantitative engine…"):
        q              = QuantCore.solve_metrics(raw)
        ai_preds, conf = AIModel.predict_trajectory(raw)
        mc_paths       = QuantCore.monte_carlo(raw, sims=mc_sims, horizon=30)

    if q is None:
        st.error("Quantitative engine returned no results. Try a different ticker.")
        return

    risk_name, risk_color, risk_emoji = risk_label(q["sharpe"], q["max_dd"], q["ann_vol"])
    lp      = float(raw["Close"].iloc[-1])
    is_bull = lp > float(raw["Close"].rolling(50).mean().iloc[-1])

    # ── TAB 1: DASHBOARD ─────────────────────────────────────
    with tabs[1]:
        st.markdown(
            f"<p style='font-size:0.78rem; color:#475569; margin-bottom:4px;'>"
            f"Data source: {status_badge(status)} &nbsp;&nbsp; "
            f"Last updated: {raw.index[-1].strftime('%d %b %Y')} &nbsp;&nbsp; "
            f"Records: {len(raw):,} trading days</p>",
            unsafe_allow_html=True)
        quick_stats_bar(raw, ticker)

        col_L, col_R = st.columns([2.1, 0.9])
        with col_L:
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            chart_candlestick(raw, ai_preds, conf if conf else [], window, show_bb)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_R:
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            section("MARKET MOOD")
            mood_color = "#39FF14" if is_bull else "#FF3131"
            mood_word  = "BULLISH"  if is_bull else "BEARISH"
            mood_icon  = "📈"       if is_bull else "📉"
            st.markdown(f"<h1 style='color:{mood_color}; font-family:Orbitron; font-size:2.2rem; margin:0;'>{mood_icon} {mood_word}</h1>",
                        unsafe_allow_html=True)
            st.markdown(f"<p style='color:#94A3B8; font-size:0.82rem; margin-top:6px;'>Price <b>${lp:,.2f}</b> is "
                        f"{'above' if is_bull else 'below'} its 50-day average.</p>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            section("RISK PROFILE")
            st.markdown(f"<p style='color:{risk_color}; font-family:Orbitron; font-size:1.4rem; font-weight:900; margin:4px 0;'>{risk_emoji} {risk_name}</p>",
                        unsafe_allow_html=True)
            st.markdown(f"<p style='color:#94A3B8; font-size:0.8rem;'>Ann. Vol: <b>{q['ann_vol']*100:.1f}%</b> · Max DD: <b>{q['max_dd']*100:.1f}%</b></p>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            section("ALPHA XP")
            st.metric("SHARPE RATIO",   f"{q['sharpe']:.2f}",  help=GLOSSARY["Sharpe"])
            st.metric("SORTINO RATIO",  f"{q['sortino']:.2f}", help=GLOSSARY["Sortino"])
            st.metric("ALPHA (Annual)", f"{q['alpha']*100:.2f}%", help=GLOSSARY["Alpha"])
            st.metric("BETA",           f"{q['beta']:.3f}",    help=GLOSSARY["Beta"])
            sharpe_progress = min(100, max(0, int(q["sharpe"] * 30)))
            st.progress(sharpe_progress, text=f"Sharpe Level: {q['sharpe']:.2f} / 3.0")
            st.markdown("</div>", unsafe_allow_html=True)

        # CSV Export
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("📥 EXPORT RAW DATA")
        csv_buf = io.StringIO()
        raw.to_csv(csv_buf)
        st.download_button(
            label=f"⬇ Download {ticker} OHLCV Data (.csv)",
            data=csv_buf.getvalue(),
            file_name=f"{ticker}_stockpulse_data.csv",
            mime="text/csv",
        )
        explain("Download the full historical OHLCV dataset used in this analysis for your own research.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: TECHNICALS ────────────────────────────────────
    with tabs[2]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        chart_macd(raw)
        st.markdown("</div>", unsafe_allow_html=True)

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            chart_rsi(raw)
            st.markdown("</div>", unsafe_allow_html=True)
        with tc2:
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            chart_fibonacci(raw, q["fibs"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        chart_return_dist(q["rets"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: MONTE CARLO ───────────────────────────────────
    with tabs[3]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        chart_monte_carlo(mc_paths, lp)
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("📖 What is Monte Carlo simulation?"):
            st.markdown("Runs **hundreds of 'what-if' futures** for your stock using the same historical return "
                        "and volatility profile.\n\n- **Narrow fan** = low volatility (more predictable)\n"
                        "- **Wide fan** = high volatility (high risk & reward)\n"
                        "- **Yellow median** = most likely single path")

    # ── TAB 4: HYPOTHESIS LAB ────────────────────────────────
    with tabs[4]:
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            section("NORMALITY HYPOTHESIS")
            if q["is_normal"]:
                st.success("✅ Accepted — Returns follow a stable distribution")
                explain("Returns are roughly bell-shaped. Standard models work well here.")
            else:
                st.warning("⚠️ Rejected — Returns have fat tails / irregular distribution")
                explain("Extreme price moves happen more often than 'normal'. Standard risk models may underestimate danger.")
            st.markdown(f"**Skewness:** `{q['skew']:.3f}`")
            st.markdown(f"**Kurtosis:** `{q['kurt']:.3f}`")
            st.caption(GLOSSARY["Skewness"])
            st.markdown("</div>", unsafe_allow_html=True)
        with hc2:
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            section("STATIONARITY HYPOTHESIS")
            if q["is_stationary"]:
                st.info("📊 Accepted — Variance is stable over time")
                explain("The stock's volatility has remained consistent. Predictive models tend to be more reliable.")
            else:
                st.warning("⚠️ Rejected — Variance is shifting over time")
                explain("Volatility has changed significantly — common during earnings, macro events, or sector rotation.")
            st.markdown(f"**Market Correlation:** `{q['corr']:.3f}`")
            st.caption("How closely this stock tracks the broader market.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("5-FACTOR SCORECARD")
        factors = [
            ("Sharpe",      q["sharpe"],   q["sharpe"] > 1.0,     f"{q['sharpe']:.2f}"),
            ("Alpha",       q["alpha"],    q["alpha"] > 0,         f"{q['alpha']*100:.2f}%"),
            ("Beta",        q["beta"],     0.5 < q["beta"] < 1.5,  f"{q['beta']:.3f}"),
            ("Max DD",      q["max_dd"],   q["max_dd"] > -0.25,    f"{q['max_dd']*100:.1f}%"),
            ("Mkt Corr",    q["corr"],     True,                   f"{q['corr']:.3f}"),
        ]
        for col, (name, val, good, display) in zip(st.columns(5), factors):
            col.metric(name, display, delta="✅" if good else "⚠️",
                       delta_color="normal" if good else "inverse")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 5: RISK MATRIX ───────────────────────────────────
    with tabs[5]:
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            section("LOSS METRICS")
            st.metric("VaR (Daily, 95%)", f"{q['var_95']*100:.3f}%",   help=GLOSSARY["VaR 95%"])
            explain(f"On the worst 5% of days, losses ≥ {abs(q['var_95'])*100:.2f}%. "
                    f"For $10,000 position = ${abs(q['var_95'])*10000:,.0f}.")
            st.metric("CVaR (95%)",       f"{q['cvar_95']*100:.3f}%",  help=GLOSSARY["CVaR 95%"])
            explain(f"Average loss on those extreme bad days = {abs(q['cvar_95'])*100:.2f}%.")
            st.markdown("</div>", unsafe_allow_html=True)
        with rc2:
            st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
            section("RETURN STATISTICS")
            st.metric("Annualised Return",     f"{q['ann_ret']*100:.2f}%")
            st.metric("Annualised Volatility", f"{q['ann_vol']*100:.2f}%")
            st.metric("Max Drawdown",          f"{q['max_dd']*100:.2f}%", help=GLOSSARY["Max Drawdown"])
            st.metric("Sortino Ratio",         f"{q['sortino']:.2f}",     help=GLOSSARY["Sortino"])
            st.metric("Calmar Ratio",          f"{q['calmar']:.2f}",      help=GLOSSARY["Calmar"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        chart_drawdown(q["drawdown_series"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        chart_rolling_sharpe(q["roll_sharpe"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Risk-Return scatter
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("RISK ↔ RETURN LANDSCAPE")
        np.random.seed(99)
        n_peers    = 15
        peer_rets  = np.random.normal(q["ann_ret"], 0.06, n_peers)
        peer_vols  = np.abs(np.random.normal(q["ann_vol"], 0.05, n_peers))
        peer_names = [f"Peer {i+1}" for i in range(n_peers)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=peer_vols, y=peer_rets, mode="markers+text",
                                 text=peer_names, textposition="top center",
                                 marker=dict(size=8, color="#1E293B",
                                             line=dict(color="#475569", width=1)), name="Peers"))
        fig.add_trace(go.Scatter(x=[q["ann_vol"]], y=[q["ann_ret"]],
                                 mode="markers+text", text=[ticker], textposition="top center",
                                 marker=dict(size=16, color="#00FFFF", symbol="star"), name=ticker))
        fig.update_layout(**CHART_LAYOUT, height=320,
                          xaxis_title="Annual Volatility", yaxis_title="Annual Return")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        explain("Ideal: upper-left (high return, low risk). Cyan star = selected stock.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Price-Target Calculator
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("🎯 PRICE-TARGET CALCULATOR")
        pt1, pt2, pt3 = st.columns(3)
        entry_price  = pt1.number_input("Entry Price ($)",  value=float(round(lp, 2)), step=0.5)
        target_price = pt2.number_input("Target Price ($)", value=float(round(lp * 1.10, 2)), step=0.5)
        stop_price   = pt3.number_input("Stop-Loss Price ($)", value=float(round(lp * 0.95, 2)), step=0.5)
        reward = target_price - entry_price
        risk   = entry_price - stop_price
        rr     = reward / risk if risk > 0 else 0
        rr_col = "#39FF14" if rr >= 2 else "#FFD700" if rr >= 1 else "#FF3131"
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Potential Gain",  f"${reward:,.2f} ({reward/entry_price*100:.1f}%)")
        pm2.metric("Potential Loss",  f"${risk:,.2f} ({risk/entry_price*100:.1f}%)")
        pm3.metric("Risk:Reward",     f"1 : {rr:.2f}")
        explain(f"A 1:{rr:.1f} R:R ratio means for every $1 risked you stand to gain ${rr:.2f}. "
                f"Professionals generally look for at least 1:2 before entering a trade.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 6: COMPARE ───────────────────────────────────────
    with tabs[6]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("MULTI-STOCK COMPARISON")
        default_peers = "AAPL,MSFT,GOOGL"
        peer_input = st.text_input("Enter tickers to compare (comma-separated)",
                                   default_peers, key="compare_tickers",
                                   help="e.g. AAPL,MSFT,TSLA — max 5 tickers")
        comp_tickers = [t.strip().upper() for t in peer_input.split(",") if t.strip()][:5]
        comp_tickers = [ticker] + [t for t in comp_tickers if t != ticker]

        dfs_dict, metrics_dict = {}, {}
        with st.spinner("Loading comparison data…"):
            for tkr in comp_tickers[:5]:
                df_c, _ = DataEngine.fetch_validated_payload(tkr, api_key)
                if df_c is not None and len(df_c) >= 60:
                    dfs_dict[tkr]    = df_c
                    m_c = QuantCore.solve_metrics(df_c)
                    if m_c:
                        metrics_dict[tkr] = m_c

        if len(dfs_dict) > 1:
            # Align to common date range
            common_start = max(df.index[0] for df in dfs_dict.values())
            aligned = {k: v[v.index >= common_start] for k, v in dfs_dict.items()}
            chart_comparison(aligned)
            st.markdown("</div>", unsafe_allow_html=True)

            if len(metrics_dict) > 1:
                st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                chart_radar(metrics_dict)
                st.markdown("</div>", unsafe_allow_html=True)

                # Comparison table
                st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
                section("SIDE-BY-SIDE METRICS TABLE")
                rows = []
                for tkr, m in metrics_dict.items():
                    rows.append({
                        "Ticker": tkr,
                        "Ann. Return": f"{m['ann_ret']*100:.2f}%",
                        "Ann. Vol":    f"{m['ann_vol']*100:.2f}%",
                        "Sharpe":      f"{m['sharpe']:.2f}",
                        "Sortino":     f"{m['sortino']:.2f}",
                        "Max DD":      f"{m['max_dd']*100:.1f}%",
                        "Calmar":      f"{m['calmar']:.2f}",
                        "Beta":        f"{m['beta']:.3f}",
                    })
                st.dataframe(pd.DataFrame(rows).set_index("Ticker"),
                             use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Add at least one more ticker in the field above to compare.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 7: WATCHLIST ─────────────────────────────────────
    with tabs[7]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("MY WATCHLIST")

        if "watchlist" not in st.session_state:
            st.session_state["watchlist"] = ["AAPL", "TSLA", "MSFT", "AMZN"]

        wl_add = st.text_input("➕ Add ticker to watchlist", key="wl_add_input",
                                placeholder="e.g. NVDA").upper()
        wa1, wa2 = st.columns([1, 3])
        with wa1:
            if st.button("Add", key="wl_add_btn") and wl_add:
                if wl_add not in st.session_state["watchlist"]:
                    st.session_state["watchlist"].append(wl_add)
                    st.rerun()

        st.divider()
        with st.spinner("Loading watchlist data…"):
            for wt in list(st.session_state["watchlist"]):
                wdf, _ = DataEngine.fetch_validated_payload(wt, api_key)
                if wdf is not None and len(wdf) >= 10:
                    wd1, wd5, _ = pct_delta(wdf)
                    wlp  = float(wdf["Close"].iloc[-1])
                    wqm  = QuantCore.solve_metrics(wdf)
                    wsharpe = f"{wqm['sharpe']:.2f}" if wqm else "—"
                    wdd  = f"{wqm['max_dd']*100:.1f}%" if wqm else "—"
                    d1_col = "#39FF14" if wd1 >= 0 else "#FF3131"
                    d5_col = "#39FF14" if wd5 >= 0 else "#FF3131"
                    wcols = st.columns([1.2, 1, 1, 1, 1, 0.5])
                    wcols[0].markdown(f"<b style='color:#00FFFF; font-family:Orbitron;'>{wt}</b>", unsafe_allow_html=True)
                    wcols[1].markdown(f"<span style='color:#FFF;'>${wlp:,.2f}</span>", unsafe_allow_html=True)
                    wcols[2].markdown(f"<span style='color:{d1_col};'>{wd1:+.2f}% 1D</span>", unsafe_allow_html=True)
                    wcols[3].markdown(f"<span style='color:{d5_col};'>{wd5:+.2f}% 5D</span>", unsafe_allow_html=True)
                    wcols[4].markdown(f"<span style='color:#94A3B8; font-size:0.8rem;'>Sharpe {wsharpe} | DD {wdd}</span>", unsafe_allow_html=True)
                    if wcols[5].button("✕", key=f"rm_{wt}"):
                        st.session_state["watchlist"].remove(wt)
                        st.rerun()
                    st.markdown("<hr style='border-color:#1E293B; margin:4px 0;'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 8: CALENDAR ──────────────────────────────────────
    with tabs[8]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section(f"EARNINGS & EVENTS CALENDAR — {ticker}")
        events = QuantCore.earnings_calendar(ticker)
        for ev in events:
            days = ev["Days Away"]
            urgency = "#FF3131" if days <= 14 else "#FFD700" if days <= 45 else "#39FF14"
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:16px; padding:12px 0;
                        border-bottom:1px solid #1E293B;'>
                <div style='background:{urgency}22; border:1px solid {urgency}; border-radius:8px;
                            padding:6px 12px; font-family:Orbitron; font-size:0.7rem; color:{urgency};
                            min-width:80px; text-align:center;'>{ev["Days Away"]}d</div>
                <div>
                    <div style='color:#FFF; font-weight:700;'>{ev["Event"]}</div>
                    <div style='color:#475569; font-size:0.8rem;'>{ev["Date"]} &nbsp;·&nbsp; Est. EPS: {ev["Est. EPS"]}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        explain("Simulated events for demonstration. With a live API key, real earnings dates can be fetched from Finnhub's /earnings endpoint.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 9: SENTIMENT ─────────────────────────────────────
    with tabs[9]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section(f"NEWS SENTIMENT SCORE — {ticker}")
        score, headlines = QuantCore.sentiment_score(ticker)
        score_color = "#39FF14" if score > 60 else "#FFD700" if score > 40 else "#FF3131"
        score_label = "BULLISH" if score > 60 else "NEUTRAL" if score > 40 else "BEARISH"
        st.markdown(f"""
        <div style='text-align:center; padding:20px 0;'>
            <div style='font-family:Orbitron; font-size:3rem; font-weight:900; color:{score_color};'>{score:.0f}</div>
            <div style='font-family:Orbitron; font-size:0.8rem; color:{score_color}; letter-spacing:4px;'>{score_label} SENTIMENT</div>
        </div>
        <div class='sentiment-bar' style='margin:10px 0 20px 0;'>
            <div style='position:absolute; left:{score}%; top:-4px; width:4px; height:20px;
                        background:#FFF; border-radius:2px;'></div>
        </div>""", unsafe_allow_html=True)

        section("SIMULATED HEADLINE FEED")
        for headline, tone in headlines:
            tone_col = "#39FF14" if tone == "Bullish" else "#FF3131" if tone == "Bearish" else "#FFD700"
            st.markdown(f"""
            <div style='display:flex; align-items:center; justify-content:space-between;
                        padding:10px 0; border-bottom:1px solid #1E293B;'>
                <span style='color:#CBD5E1; font-size:0.88rem;'>📰 {headline}</span>
                <span style='color:{tone_col}; font-family:Orbitron; font-size:0.65rem;
                             background:{tone_col}18; border:1px solid {tone_col};
                             border-radius:20px; padding:2px 10px;'>{tone}</span>
            </div>""", unsafe_allow_html=True)
        explain("Sentiment score and headlines are simulated for educational use. "
                "In production, connect to a news NLP API (e.g. Finnhub /news-sentiment) for real scores.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 10: PORTFOLIO ────────────────────────────────────
    with tabs[10]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("OPTIMAL ASSET LOADOUT")
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("ASSET WEIGHT", "40%", "+2% REBAL")
        oc2.metric("CASH BUFFER",  "30%", "-1% FEE")
        oc3.metric("BTC RATIO",    "30%", "+5% XP")
        explain("Illustrative weights. Real portfolios use mean-variance (Markowitz) or risk-parity optimisation.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("POSITION SIZING CALCULATOR")
        port_size = st.number_input("Portfolio Size ($)", 1000, 10_000_000, 10_000, 1000)
        risk_pct  = st.slider("Max Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5,
                              help="Classic '2% rule'.")
        stop_pct  = st.slider("Stop-Loss Distance (%)", 1.0, 15.0, 5.0, 0.5)
        max_loss   = port_size * (risk_pct / 100)
        pos_size   = max_loss / (stop_pct / 100)
        num_shares = int(pos_size / lp)
        p1, p2, p3 = st.columns(3)
        p1.metric("Max $ Risk",         f"${max_loss:,.0f}")
        p2.metric("Recommended Pos.",   f"${pos_size:,.0f}")
        p3.metric("Approx. Shares",     str(num_shares))
        explain(f"If {ticker} drops {stop_pct}% and hits your stop, you lose exactly "
                f"${max_loss:,.0f} — {risk_pct}% of your portfolio.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 11: TRADE JOURNAL ────────────────────────────────
    with tabs[11]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("📋 TRADE JOURNAL")

        if "journal" not in st.session_state:
            st.session_state["journal"] = []

        with st.form("add_trade_form"):
            jc1, jc2, jc3, jc4, jc5 = st.columns(5)
            j_ticker = jc1.text_input("Ticker", value=ticker)
            j_dir    = jc2.selectbox("Direction", ["Long", "Short"])
            j_entry  = jc3.number_input("Entry $", value=float(round(lp, 2)), step=0.5)
            j_exit   = jc4.number_input("Exit $",  value=float(round(lp * 1.05, 2)), step=0.5)
            j_size   = jc5.number_input("Shares",  value=10, step=1)
            j_notes  = st.text_input("Notes (optional)", placeholder="e.g. Breakout above MA50")
            submitted = st.form_submit_button("➕ Log Trade")
            if submitted:
                pnl = (j_exit - j_entry) * j_size if j_dir == "Long" else (j_entry - j_exit) * j_size
                st.session_state["journal"].append({
                    "Date":      datetime.today().strftime("%d %b %Y"),
                    "Ticker":    j_ticker.upper(),
                    "Direction": j_dir,
                    "Entry":     j_entry,
                    "Exit":      j_exit,
                    "Shares":    j_size,
                    "P&L":       round(pnl, 2),
                    "Notes":     j_notes,
                })
                st.success(f"Trade logged! P&L: ${pnl:,.2f}")

        if st.session_state["journal"]:
            jdf = pd.DataFrame(st.session_state["journal"])
            total_pnl  = jdf["P&L"].sum()
            win_rate   = (jdf["P&L"] > 0).mean() * 100
            best_trade = jdf["P&L"].max()
            worst_trade= jdf["P&L"].min()

            jm1, jm2, jm3, jm4 = st.columns(4)
            jm1.metric("Total P&L",   f"${total_pnl:,.2f}")
            jm2.metric("Win Rate",    f"{win_rate:.1f}%")
            jm3.metric("Best Trade",  f"${best_trade:,.2f}")
            jm4.metric("Worst Trade", f"${worst_trade:,.2f}")

            # Equity curve
            eq_curve = jdf["P&L"].cumsum()
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=eq_curve, mode="lines+markers",
                                        line=dict(color="#00FFFF", width=2),
                                        marker=dict(color="#00FFFF", size=6),
                                        fill="tozeroy", fillcolor="rgba(0,255,255,0.06)",
                                        name="Cumulative P&L"))
            fig_eq.add_hline(y=0, line_dash="dot", line_color="#FF3131")
            fig_eq.update_layout(**CHART_LAYOUT, height=200, yaxis_title="Cumulative P&L ($)")
            section("EQUITY CURVE")
            st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})

            st.dataframe(jdf.set_index("Date"), use_container_width=True)

            if st.button("🗑 Clear Journal"):
                st.session_state["journal"] = []
                st.rerun()
        else:
            st.info("No trades logged yet. Use the form above to record your first trade.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 12: COMMUNITY ────────────────────────────────────
    with tabs[12]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("COMMUNITY PULSE")
        feed = [
            ("@AlphaTrader",  "📍", "Seeing massive resistance at Fibonacci 61.8%. Watching closely.",    "2h ago"),
            ("@QuantSoham",   "🧠", "Hypothesis accepted for trend stationarity. Maintaining position.",  "4h ago"),
            ("@NeonHedge",    "💚", "AI Pulse looking green for next 7 days. Confidence band tight.",      "5h ago"),
            ("@RiskReaper",   "⚠️", "VaR spiking post-FOMC. Reducing leverage to 1.5×.",                  "8h ago"),
            ("@MomentumMike", "🚀", "MACD crossed above signal on the daily — bullish continuation.",      "12h ago"),
        ]
        for handle, icon, text, time_ago in feed:
            st.markdown(f"""<div style='border-bottom:1px solid #1E293B; padding:12px 0;'>
                    <span style='color:#00FFFF; font-weight:700;'>{handle}</span>
                    &nbsp;<span style='color:#475569; font-size:0.75rem;'>{time_ago}</span><br>
                    <span style='font-size:0.88rem; color:#CBD5E1;'>{icon} {text}</span>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 13: GLOSSARY ─────────────────────────────────────
    with tabs[13]:
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        section("FINANCIAL TERMS GLOSSARY")
        st.markdown("<p style='color:#94A3B8; font-size:0.85rem; margin-bottom:16px;'>"
                    "Every term used in StockPulse AI, explained in plain English.</p>",
                    unsafe_allow_html=True)
        for term, definition in GLOSSARY.items():
            with st.expander(f"📘 {term}"):
                st.markdown(f"> {definition}")
        st.divider()
        section("COMMON QUESTIONS")
        faqs = [
            ("What does 'simulated data' mean?",
             "When no API key is entered, StockPulse AI generates realistic synthetic price data. "
             "All calculations work the same — great for learning, not for real trading decisions."),
            ("Is this financial advice?",
             "No. StockPulse AI is an educational tool. Always do your own research and consult a qualified "
             "financial advisor before making investment decisions."),
            ("How accurate is the AI prediction?",
             "The 7-day forecast uses a Random Forest model on technical indicators. "
             "It captures trends but not news events. Treat it as one data point."),
            ("What is a good Sharpe ratio?",
             "Above 1.0 is good, above 2.0 is excellent, above 3.0 is exceptional."),
            ("What's the difference between Sharpe and Sortino?",
             "Sharpe penalises all volatility; Sortino only penalises downside (bad) volatility. "
             "Sortino is often a better measure for trend-following strategies."),
        ]
        for q_text, a_text in faqs:
            with st.expander(f"❓ {q_text}"):
                st.markdown(a_text)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────
    st.write("")
    st.divider()
    st.markdown(
        f"<center><p style='color:#1E293B; font-family:Orbitron; font-size:0.65rem; letter-spacing:3px;'>"
        f"STOCKPULSE AI TERMINAL // v{VERSION} // FOR EDUCATIONAL USE ONLY // NOT FINANCIAL ADVICE"
        f"</p></center>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()