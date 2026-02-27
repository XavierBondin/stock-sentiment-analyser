import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# Streamlit lets you configure the page title,
# icon, and layout before anything else renders.
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sentiment Analyser",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# Streamlit supports injecting custom CSS via
# st.markdown with unsafe_allow_html=True.
# ─────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #07070f;
            color: #e0e0f0;
        }
        .main { background-color: #07070f; }
        .block-container { padding-top: 2rem; }

        .big-title {
            font-family: 'Bebas Neue', cursive;
            font-size: 64px;
            letter-spacing: 3px;
            line-height: 1;
            margin-bottom: 4px;
        }
        .green { color: #4ade80; }
        .red { color: #f87171; }
        .yellow { color: #facc15; }
        .grey { color: #888; }

        .card {
            background: #0e0e1a;
            border: 1px solid #1e1e35;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }
        .headline-positive {
            border-left: 3px solid #4ade80;
            background: #0e0e1a;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .headline-negative {
            border-left: 3px solid #f87171;
            background: #0e0e1a;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .headline-neutral {
            border-left: 3px solid #facc15;
            background: #0e0e1a;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .tag {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
        }
        .stTextInput > div > div > input {
            background-color: #0e0e1a !important;
            border: 1px solid #1e1e35 !important;
            color: #e0e0f0 !important;
            border-radius: 8px !important;
        }
        .stButton > button {
            background: #4ade80;
            color: #07070f;
            font-family: 'Bebas Neue', cursive;
            font-size: 18px;
            letter-spacing: 2px;
            border: none;
            border-radius: 8px;
            padding: 10px 28px;
            width: 100%;
        }
        .stButton > button:hover {
            background: #22c55e;
            color: #07070f;
        }
        div[data-testid="metric-container"] {
            background: #0e0e1a;
            border: 1px solid #1e1e35;
            border-radius: 10px;
            padding: 16px;
        }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FUNCTIONS
# These are the core building blocks of the app.
# Each function does one specific job.
# ─────────────────────────────────────────────

def fetch_headlines(query: str, api_key: str) -> list[dict]:
    """
    Fetches recent news headlines for a given stock/company
    using the NewsAPI (newsapi.org - free tier available).

    Returns a list of article dicts with 'title' and 'source'.
    If no API key is provided, returns sample headlines for demo.
    """
    if not api_key:
        # Demo mode — return sample headlines so the app still works
        return [
            {"title": f"{query} reports record quarterly earnings, beating analyst expectations", "source": "Bloomberg"},
            {"title": f"{query} faces regulatory scrutiny over data practices", "source": "Reuters"},
            {"title": f"Analysts upgrade {query} stock to 'buy' amid strong growth outlook", "source": "CNBC"},
            {"title": f"{query} announces major layoffs as part of restructuring plan", "source": "Financial Times"},
            {"title": f"{query} expands into new markets with strategic acquisition", "source": "Wall Street Journal"},
            {"title": f"Investors cautious as {query} misses revenue targets", "source": "MarketWatch"},
            {"title": f"{query} CEO confident about long-term growth trajectory", "source": "Forbes"},
            {"title": f"Supply chain concerns weigh on {query} outlook", "source": "The Guardian"},
        ]

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 15,
        "apiKey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        articles = data.get("articles", [])
        return [
            {"title": a["title"], "source": a["source"]["name"]}
            for a in articles
            if a.get("title") and "[Removed]" not in a["title"]
        ]
    except Exception:
        st.warning("Could not fetch live headlines. Showing demo data instead.")
        return fetch_headlines(query, api_key=None)


def score_headlines(headlines: list[dict]) -> list[dict]:
    """
    Runs each headline through VADER sentiment analysis.

    VADER (Valence Aware Dictionary and sEntiment Reasoner)
    is an NLP model that scores text from -1 (very negative)
    to +1 (very positive). It's specifically tuned for
    short, news-style text — perfect for headlines.

    The 'compound' score is the overall sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []

    for h in headlines:
        scores = analyzer.polarity_scores(h["title"])
        compound = scores["compound"]  # -1.0 to +1.0

        # Classify into buckets
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        results.append({
            "headline": h["title"],
            "source": h["source"],
            "score": round(compound, 3),
            "label": label,
            "positive_component": round(scores["pos"], 3),
            "negative_component": round(scores["neg"], 3),
            "neutral_component": round(scores["neu"], 3),
        })

    return results


def compute_overall_score(scored: list[dict]) -> float:
    """
    Computes the overall sentiment score as a simple
    average of all individual headline compound scores.
    Returns a value between -1.0 and +1.0.
    """
    if not scored:
        return 0.0
    return round(sum(h["score"] for h in scored) / len(scored), 3)


def sentiment_label(score: float) -> tuple[str, str]:
    """Returns a (label, colour) tuple based on the overall score."""
    if score > 0.2:
        return "BULLISH 📈", "#4ade80"
    elif score < -0.2:
        return "BEARISH 📉", "#f87171"
    else:
        return "NEUTRAL ➡️", "#facc15"


def make_gauge(score: float, color: str) -> go.Figure:
    """
    Creates a Plotly gauge chart for the overall sentiment score.
    Gauge goes from -1 (red) to +1 (green).
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"color": color, "size": 36}, "suffix": ""},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": "#444", "tickfont": {"color": "#444"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0e0e1a",
            "bordercolor": "#1e1e35",
            "steps": [
                {"range": [-1, -0.2], "color": "#1a0a0a"},
                {"range": [-0.2, 0.2], "color": "#1a1a0a"},
                {"range": [0.2, 1], "color": "#0a1a0a"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=20, b=0, l=20, r=20),
        paper_bgcolor="#0e0e1a",
        font={"color": "#e0e0f0"}
    )
    return fig


def make_bar_chart(scored: list[dict]) -> go.Figure:
    """
    Creates a bar chart showing the count of
    positive, neutral, and negative headlines.
    """
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for h in scored:
        counts[h["label"].capitalize()] += 1

    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=["#4ade80", "#facc15", "#f87171"],
        marker_line_width=0,
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="#0e0e1a",
        plot_bgcolor="#0e0e1a",
        font={"color": "#888"},
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, tickfont={"color": "#444"}),
    )
    return fig


# ─────────────────────────────────────────────
# UI — HEADER
# st.markdown lets us inject raw HTML.
# st.columns splits the page into side-by-side sections.
# ─────────────────────────────────────────────
st.markdown("""
    <div class="big-title">Stock <span class="green">Sentiment</span><br>Analyser</div>
    <p class="grey" style="margin-bottom: 28px; font-size: 14px;">
        NLP-powered news sentiment analysis · Powered by VADER · Built with Python & Streamlit
    </p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UI — SIDEBAR (settings)
# st.sidebar puts content in a collapsible panel
# on the left. Good for settings/config.
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    api_key = st.text_input(
        "NewsAPI Key (optional)",
        type="password",
        help="Get a free key at newsapi.org for live headlines. Leave blank for demo mode."
    )
    st.markdown("""
        <small class="grey">
        Leave blank to run in <b>demo mode</b> with sample headlines.<br><br>
        Get a free API key at <a href="https://newsapi.org" target="_blank" style="color:#4ade80">newsapi.org</a>
        to fetch live news.
        </small>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <small class="grey">
        <b>How it works:</b><br>
        1. Fetches recent headlines for your query<br>
        2. Scores each one using VADER NLP<br>
        3. Aggregates into an overall sentiment signal<br><br>
        <b>Score ranges:</b><br>
        🟢 > 0.2 = Bullish<br>
        🟡 -0.2 to 0.2 = Neutral<br>
        🔴 < -0.2 = Bearish
        </small>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UI — SEARCH INPUT
# st.columns([x, y]) creates two columns with
# relative widths. We use 4:1 for input + button.
# ─────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("", placeholder="Enter stock ticker or company name — e.g. NVDA, Tesla, Apple...")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("ANALYSE")

# ─────────────────────────────────────────────
# MAIN LOGIC
# st.spinner shows a loading indicator while
# the code inside it is executing.
# ─────────────────────────────────────────────
if run and query:
    with st.spinner(f"Scanning headlines for {query.upper()}..."):
        headlines = fetch_headlines(query, api_key)
        scored = score_headlines(headlines)
        overall = compute_overall_score(scored)
        label, color = sentiment_label(overall)

    # ── TOP METRICS ROW ──
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Score", f"{overall:+.3f}")
    m2.metric("Signal", label.split()[0])
    m3.metric("Headlines Analysed", len(scored))
    m4.metric("Positive / Negative",
              f"{sum(1 for h in scored if h['label']=='positive')} / {sum(1 for h in scored if h['label']=='negative')}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GAUGE + BAR CHART ──
    g_col, b_col = st.columns(2)
    with g_col:
        st.markdown('<div class="card"><p style="font-size:10px;letter-spacing:3px;color:#444;text-transform:uppercase">Sentiment Gauge</p>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(overall, color), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b_col:
        st.markdown('<div class="card"><p style="font-size:10px;letter-spacing:3px;color:#444;text-transform:uppercase">Breakdown</p>', unsafe_allow_html=True)
        st.plotly_chart(make_bar_chart(scored), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── HEADLINES ──
    st.markdown('<p style="font-size:10px;letter-spacing:3px;color:#444;text-transform:uppercase;margin-top:8px">Analysed Headlines · NLP Scored</p>', unsafe_allow_html=True)

    for h in scored:
        css_class = f"headline-{h['label']}"
        score_color = "#4ade80" if h["label"] == "positive" else "#f87171" if h["label"] == "negative" else "#facc15"
        st.markdown(f"""
            <div class="{css_class}">
                <div style="font-size:13px;color:#c8c8d8;margin-bottom:8px">{h['headline']}</div>
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <span style="background:{score_color}22;color:{score_color};font-size:10px;padding:2px 10px;border-radius:20px;font-weight:700;letter-spacing:2px">
                            {h['label'].upper()}
                        </span>
                        <span style="color:#444;font-size:11px;margin-left:8px">{h['source']}</span>
                    </div>
                    <span style="color:{score_color};font-size:13px;font-weight:600">{h['score']:+.3f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── RAW DATA TABLE ──
    with st.expander("📊 View raw data as table"):
        df = pd.DataFrame(scored)
        st.dataframe(df, use_container_width=True)

    st.markdown("""
        <p style="color:#2a2a3a;font-size:11px;text-align:center;margin-top:24px;letter-spacing:1px">
        Sentiment scored via VADER NLP · For educational & research purposes only · Not financial advice
        </p>
    """, unsafe_allow_html=True)

elif run and not query:
    st.warning("Please enter a stock ticker or company name.")
