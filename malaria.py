"""
Malaria Seasonal Variation & High-Risk Period Prediction
Meru University of Science and Technology — TECH TITANS
BSc Data Science Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score
)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Malaria Predictor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme Colors ───────────────────────────────────────────────────────────────
GREEN  = "#2E7D32"
LGREEN = "#66BB6A"
RED    = "#C62828"
ORANGE = "#E65100"
BLUE   = "#1565C0"
GOLD   = "#F9A825"
BG     = "#F1F8E9"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .hero {{
    background: linear-gradient(135deg, {GREEN} 0%, #1B5E20 100%);
    padding: 2rem 2.5rem; border-radius: 14px; color: white;
    text-align: center; margin-bottom: 1.5rem;
  }}
  .hero h1 {{ margin: 0; font-size: 2.2rem; }}
  .hero p  {{ margin: 0.4rem 0 0; opacity: 0.88; font-size: 1rem; }}
  .kpi-card {{
    background: white; border-left: 5px solid {GREEN};
    border-radius: 10px; padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 0.5rem;
  }}
  .kpi-val  {{ font-size: 1.8rem; font-weight: 800; color: {GREEN}; }}
  .kpi-lbl  {{ font-size: 0.82rem; color: #555; margin-top: 2px; }}
  .risk-high {{
    background: #FFEBEE; border-left: 5px solid {RED};
    border-radius: 10px; padding: 1rem 1.4rem; margin: 0.5rem 0;
  }}
  .risk-low {{
    background: #E8F5E9; border-left: 5px solid {GREEN};
    border-radius: 10px; padding: 1rem 1.4rem; margin: 0.5rem 0;
  }}
  .sec-hdr {{
    color: {GREEN}; border-bottom: 2px solid {LGREEN};
    padding-bottom: 4px; margin: 1.4rem 0 0.8rem;
    font-size: 1.15rem; font-weight: 700;
  }}
  .stButton > button {{
    background: linear-gradient(135deg, {GREEN}, #1B5E20);
    color: white; font-weight: 700; border: none;
    border-radius: 8px; padding: 0.55rem 1.8rem; width: 100%;
  }}
  .pill {{
    display: inline-block; border-radius: 20px; padding: 3px 12px;
    font-size: 0.78rem; font-weight: 700; margin-right: 6px; color: white;
  }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(raw_bytes):
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
    df = df.sort_values("Date").reset_index(drop=True)

    # Temporal features
    df["Month"]    = df["Date"].dt.month
    df["Year"]     = df["Date"].dt.year
    df["Quarter"]  = df["Date"].dt.quarter
    df["DayOfYear"]= df["Date"].dt.dayofyear
    df["MonthName"]= df["Date"].dt.strftime("%b")

    # Kenyan seasons
    def season(m):
        if m in [3, 4, 5]:   return "Long Rains (Mar–May)"
        if m in [10, 11, 12]: return "Short Rains (Oct–Dec)"
        if m in [6, 7, 8, 9]: return "Dry Season (Jun–Sep)"
        return "Dry Season (Jan–Feb)"
    df["Season"] = df["Month"].map(season)

    # Lagged features (7-day proxy = 1 row shift per county)
    for col in ["Rainfall_mm", "Temperature_avg", "Humidity_percent", "Mosquito_density"]:
        df[f"{col}_lag1"] = df.groupby("County")[col].shift(1)
    df.fillna(method="bfill", inplace=True)

    # Composite risk index
    scaler = StandardScaler()
    risk_feats = ["Rainfall_mm", "Mosquito_density", "Humidity_percent"]
    df["Risk_Index"] = scaler.fit_transform(df[risk_feats]).mean(axis=1)

    # Binary high-risk label (top 25% cases)
    threshold = df["Total_cases"].quantile(0.75)
    df["High_Risk"] = (df["Total_cases"] >= threshold).astype(int)
    df["Risk_Label"] = df["High_Risk"].map({1: "High Risk", 0: "Low Risk"})

    # Case fatality rate
    df["CFR"] = (df["Deaths"] / df["Total_cases"].replace(0, np.nan) * 100).fillna(0)

    return df, threshold


FEATURES = ["Rainfall_mm", "Temperature_avg", "Humidity_percent",
            "Mosquito_density", "Intervention_coverage",
            "Rainfall_mm_lag1", "Mosquito_density_lag1", "Month", "Quarter"]


@st.cache_resource
def train_models(df):
    X = df[FEATURES].copy()
    y_reg  = df["Total_cases"]
    y_cls  = df["High_Risk"]

    X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _,    _,    yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)

    # Regression — Random Forest (primary, as per paper)
    rf_reg = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rf_reg.fit(X_tr, yr_tr)
    yr_pred = rf_reg.predict(X_te)

    reg_metrics = {
        "MAE":  round(mean_absolute_error(yr_te, yr_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(yr_te, yr_pred)), 2),
        "R2":   round(r2_score(yr_te, yr_pred), 4),
    }

    # Classification — Random Forest (binary high-risk)
    rf_cls = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf_cls.fit(X_tr, yc_tr)
    yc_pred   = rf_cls.predict(X_te)
    yc_proba  = rf_cls.predict_proba(X_te)[:, 1]

    cls_metrics = {
        "Accuracy":  round((yc_pred == yc_te).mean() * 100, 2),
        "F1":        round(f1_score(yc_te, yc_pred), 4),
        "Precision": round(precision_score(yc_te, yc_pred), 4),
        "Recall":    round(recall_score(yc_te, yc_pred), 4),
        "AUC_ROC":   round(roc_auc_score(yc_te, yc_proba), 4),
        "CM":        confusion_matrix(yc_te, yc_pred),
        "fpr":       roc_curve(yc_te, yc_proba)[0],
        "tpr":       roc_curve(yc_te, yc_proba)[1],
        "y_te":      yc_te.values,
        "y_pred":    yc_pred,
    }

    feat_imp = pd.Series(rf_reg.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return rf_reg, rf_cls, reg_metrics, cls_metrics, feat_imp, X_te, yr_te, yr_pred


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:1rem;background:{GREEN};border-radius:10px;color:white;margin-bottom:1rem;'>
      <div style='font-size:2.5rem;'>🦟</div>
      <div style='font-weight:800;font-size:1rem;'>Kenya Malaria Predictor</div>
      <div style='font-size:0.75rem;opacity:0.85;'>MUST · TECH TITANS · 2024</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    test_split  = st.slider("Test Split %", 10, 40, 20)
    n_estimators= st.slider("RF Trees", 50, 300, 150, step=50)
    risk_thresh = st.slider("High-Risk Threshold (quantile)", 0.60, 0.90, 0.75, 0.05,
                            help="Cases above this percentile are labelled High-Risk")
    st.markdown("---")
    st.caption("📄 Based on: Analyze the Seasonal Variation of Malaria Incidents in Kenya")

# ═══════════════════════════════════════════════════════════════════════════════
# GATE
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
    <div class='hero'>
      <h1>🦟 Kenya Malaria Seasonal Predictor</h1>
      <p>Analyze seasonal variation · Predict high-risk periods · Random Forest + ARIMA-style decomposition</p>
      <p style='margin-top:0.6rem;font-size:0.85rem;opacity:0.75;'>
        Meru University of Science & Technology · Dept. of Computer Science · BSc Data Science
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Upload **malaria_dataset_kenya_2022_2024.csv** in the sidebar to begin.")
    st.stop()

df, auto_threshold = load_data(uploaded.getvalue())
# Recompute threshold from slider
threshold = df["Total_cases"].quantile(risk_thresh)
df["High_Risk"]   = (df["Total_cases"] >= threshold).astype(int)
df["Risk_Label"]  = df["High_Risk"].map({1: "High Risk", 0: "Low Risk"})

rf_reg, rf_cls, reg_metrics, cls_metrics, feat_imp, X_te, yr_te, yr_pred = train_models(df)

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER + KPIs
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
  <h1>🦟 Kenya Malaria Seasonal Predictor</h1>
  <p>Seasonal variation analysis · High-risk period prediction · 2022–2024</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    ("📋 Records",      f"{len(df):,}",                            "Dataset rows"),
    ("🏥 Total Cases",  f"{df['Total_cases'].sum():,.0f}",         "Reported malaria cases"),
    ("💀 Deaths",       f"{df['Deaths'].sum():,}",                 "Malaria fatalities"),
    ("📍 Counties",     f"{df['County'].nunique()}",               "Counties tracked"),
    ("⭐ Avg Cases/Day",f"{df['Total_cases'].mean():.1f}",         "Daily county avg"),
    ("🚨 High-Risk %",  f"{df['High_Risk'].mean()*100:.1f}%",      "Above risk threshold"),
]
for col, (icon_label, val, lbl) in zip([k1,k2,k3,k4,k5,k6], kpis):
    col.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-val'>{val}</div>
      <div class='kpi-lbl'>{icon_label}<br>{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA & Seasonal Trends",
    "🗺️ Regional Analysis",
    "🌡️ Environmental Factors",
    "🤖 ML Models & Metrics",
    "🔮 Predict High-Risk Period",
    "🗂️ Data Explorer"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA & SEASONAL TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='sec-hdr'>📈 Monthly Malaria Cases — Seasonal Pattern</div>", unsafe_allow_html=True)

    monthly_avg = df.groupby("Month")["Total_cases"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_avg["MonthName"] = monthly_avg["Month"].apply(lambda x: month_names[x-1])

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor="white")
    bar_colors = [RED if m in [3,4,5,10,11,12] else LGREEN for m in monthly_avg["Month"]]
    bars = ax.bar(monthly_avg["MonthName"], monthly_avg["Total_cases"], color=bar_colors, edgecolor="white", alpha=0.9)
    ax.plot(monthly_avg["MonthName"], monthly_avg["Total_cases"], "o-", color=GREEN, linewidth=2.5, markersize=7, zorder=5)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f"{bar.get_height():.0f}", ha="center", fontsize=8.5, fontweight="bold")
    # Season shading
    for span, label, col in [([2.5,5.5],"Long Rains\n(Mar–May)","#FFCDD2"),
                               ([9.5,11.5],"Short Rains\n(Oct–Dec)","#FFCDD2"),]:
        ax.axvspan(*span, alpha=0.18, color=col)
        ax.text(np.mean(span), ax.get_ylim()[1]*0.95, label, ha="center",
                fontsize=8, color=RED, fontweight="bold")
    ax.set_ylabel("Average Total Cases"); ax.set_title("Average Monthly Malaria Cases (2022–2024)", fontweight="bold")
    ax.set_facecolor("#F9FBE7")
    legend_el = [mpatches.Patch(color=RED, label="Rainy Season (High Risk)"),
                 mpatches.Patch(color=LGREEN, label="Dry Season (Lower Risk)")]
    ax.legend(handles=legend_el, loc="upper right")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='sec-hdr'>🍂 Cases by Season</div>", unsafe_allow_html=True)
        season_avg = df.groupby("Season")["Total_cases"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        colors = [RED if "Rain" in s else LGREEN for s in season_avg.index]
        bars = ax.barh(season_avg.index, season_avg.values, color=colors, edgecolor="white", alpha=0.88)
        for bar in bars:
            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.1f}", va="center", fontsize=10, fontweight="bold")
        ax.set_xlabel("Avg Cases"); ax.set_title("Average Cases by Season", fontweight="bold")
        ax.set_facecolor("#F9FBE7"); plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown("<div class='sec-hdr'>📅 Year-over-Year Trend</div>", unsafe_allow_html=True)
        yearly = df.groupby(["Year","Month"])["Total_cases"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        palette = {2022: GREEN, 2023: ORANGE, 2024: BLUE}
        for yr in sorted(yearly["Year"].unique()):
            sub = yearly[yearly["Year"]==yr]
            ax.plot(sub["Month"], sub["Total_cases"], "o-", label=str(yr),
                    color=palette.get(yr, "gray"), linewidth=2.2, markersize=6)
        ax.set_xticks(range(1,13)); ax.set_xticklabels(month_names, fontsize=8)
        ax.set_ylabel("Avg Cases"); ax.set_title("Year-over-Year Monthly Trend", fontweight="bold")
        ax.legend(); ax.set_facecolor("#F9FBE7"); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='sec-hdr'>📦 Case Distribution by Season (Box Plot)</div>", unsafe_allow_html=True)
    season_order = ["Long Rains (Mar–May)", "Short Rains (Oct–Dec)", "Dry Season (Jun–Sep)", "Dry Season (Jan–Feb)"]
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
    data_by_season = [df[df["Season"]==s]["Total_cases"].values for s in season_order]
    bp = ax.boxplot(data_by_season, labels=[s.split(" (")[0] for s in season_order],
                    patch_artist=True, medianprops=dict(color="white", linewidth=2))
    box_colors = [RED, RED, LGREEN, LGREEN]
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_ylabel("Total Cases"); ax.set_title("Case Distribution by Season", fontweight="bold")
    ax.set_facecolor("#F9FBE7"); plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REGIONAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='sec-hdr'>🗺️ Cases by Region & County</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        region_stats = df.groupby("Region").agg(
            Avg_Cases=("Total_cases","mean"),
            Total_Cases=("Total_cases","sum"),
            Avg_Deaths=("Deaths","mean"),
            High_Risk_Pct=("High_Risk","mean")
        ).round(2).sort_values("Avg_Cases", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        bars = ax.bar(region_stats.index, region_stats["Avg_Cases"],
                      color=[RED, ORANGE, GOLD, LGREEN, BLUE][:len(region_stats)], edgecolor="white", alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f"{bar.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Avg Cases"); ax.set_title("Average Cases by Region", fontweight="bold")
        ax.set_facecolor("#F9FBE7"); plt.xticks(rotation=20, ha="right")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        county_avg = df.groupby("County")["Total_cases"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        cmap_vals = [RED if v > county_avg.quantile(0.75) else LGREEN for v in county_avg.values]
        ax.barh(county_avg.index[::-1], county_avg.values[::-1], color=cmap_vals[::-1], edgecolor="white", alpha=0.88)
        ax.set_xlabel("Avg Cases"); ax.set_title("Average Cases by County", fontweight="bold")
        ax.set_facecolor("#F9FBE7"); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='sec-hdr'>📊 Region Summary Table</div>", unsafe_allow_html=True)
    region_stats["High_Risk_Pct"] = (region_stats["High_Risk_Pct"] * 100).round(1).astype(str) + "%"
    st.dataframe(region_stats.rename(columns={
        "Avg_Cases":"Avg Cases/Day", "Total_Cases":"Total Cases",
        "Avg_Deaths":"Avg Deaths/Day", "High_Risk_Pct":"High-Risk %"
    }), use_container_width=True)

    st.markdown("<div class='sec-hdr'>🔥 Regional Seasonal Heatmap</div>", unsafe_allow_html=True)
    pivot = df.pivot_table(values="Total_cases", index="Region", columns="Month", aggfunc="mean")
    pivot.columns = month_names
    fig, ax = plt.subplots(figsize=(13, 4), facecolor="white")
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=0.4, annot_kws={"size": 9})
    ax.set_title("Avg Malaria Cases by Region & Month", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='sec-hdr'>☠️ Case Fatality Rate (CFR) by County</div>", unsafe_allow_html=True)
    cfr_county = df.groupby("County")["CFR"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor="white")
    bars = ax.bar(cfr_county.index, cfr_county.values, color=RED, alpha=0.78, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.2f}%", ha="center", fontsize=8.5)
    ax.set_ylabel("CFR (%)"); ax.set_title("Case Fatality Rate by County", fontweight="bold")
    ax.set_facecolor("#F9FBE7"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ENVIRONMENTAL FACTORS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='sec-hdr'>🌡️ Environmental Factor Analysis</div>", unsafe_allow_html=True)

    env_cols = ["Rainfall_mm", "Temperature_avg", "Humidity_percent",
                "Mosquito_density", "Intervention_coverage"]

    # Correlation matrix
    c1, c2 = st.columns(2)
    with c1:
        corr = df[["Total_cases"] + env_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = False
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                    linewidths=0.4, annot_kws={"size":9}, vmin=-1, vmax=1)
        ax.set_title("Correlation: Cases vs Environmental Factors", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        # Rainfall vs Cases scatter
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
        sc = ax.scatter(df["Rainfall_mm"], df["Total_cases"],
                        c=df["Mosquito_density"], cmap="YlOrRd", alpha=0.4, s=12)
        plt.colorbar(sc, ax=ax, label="Mosquito Density")
        z = np.polyfit(df["Rainfall_mm"], df["Total_cases"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["Rainfall_mm"].min(), df["Rainfall_mm"].max(), 100)
        ax.plot(x_line, p(x_line), "--", color=GREEN, linewidth=2, label="Trend")
        ax.set_xlabel("Rainfall (mm)"); ax.set_ylabel("Total Cases")
        ax.set_title("Rainfall vs Malaria Cases\n(colour = Mosquito Density)", fontweight="bold")
        ax.legend(); ax.set_facecolor("#F9FBE7")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Monthly environmental trends
    st.markdown("<div class='sec-hdr'>📅 Monthly Environmental Trends vs Cases</div>", unsafe_allow_html=True)
    monthly_env = df.groupby("Month")[["Total_cases","Rainfall_mm","Mosquito_density","Humidity_percent"]].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="white")
    pairs = [("Rainfall_mm","Rainfall (mm)",BLUE), ("Mosquito_density","Mosquito Density",ORANGE), ("Humidity_percent","Humidity (%)",GREEN)]
    for ax, (col, label, col_color) in zip(axes, pairs):
        ax2 = ax.twinx()
        ax.bar(month_names, monthly_env["Total_cases"], color="#C8E6C9", alpha=0.7, label="Cases")
        ax2.plot(month_names, monthly_env[col], "o-", color=col_color, linewidth=2.2,
                 markersize=6, label=label)
        ax.set_ylabel("Cases", color="#2E7D32"); ax2.set_ylabel(label, color=col_color)
        ax.set_title(f"Cases vs {label}", fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        ax.set_facecolor("#F9FBE7")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Intervention vs Cases
    st.markdown("<div class='sec-hdr'>💉 Intervention Coverage vs Cases</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="white")
    ax.scatter(df["Intervention_coverage"], df["Total_cases"],
               c=df["High_Risk"].map({1: RED, 0: LGREEN}), alpha=0.35, s=10)
    z = np.polyfit(df["Intervention_coverage"], df["Total_cases"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["Intervention_coverage"].min(), df["Intervention_coverage"].max(), 100)
    ax.plot(x_line, p(x_line), "--", color=BLUE, linewidth=2.5, label="Trend")
    legend_el = [mpatches.Patch(color=RED, label="High Risk"), mpatches.Patch(color=LGREEN, label="Low Risk")]
    ax.legend(handles=legend_el)
    ax.set_xlabel("Intervention Coverage (%)"); ax.set_ylabel("Total Cases")
    ax.set_title("Impact of Intervention Coverage on Malaria Cases", fontweight="bold")
    ax.set_facecolor("#F9FBE7"); plt.tight_layout(); st.pyplot(fig); plt.close()

    # Distribution of each env factor
    st.markdown("<div class='sec-hdr'>📊 Environmental Variable Distributions</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), facecolor="white")
    colors_env = [BLUE, ORANGE, GREEN, RED, GOLD]
    for ax, col, col_color in zip(axes, env_cols, colors_env):
        ax.hist(df[col].dropna(), bins=25, color=col_color, alpha=0.82, edgecolor="white")
        ax.axvline(df[col].mean(), color="black", linestyle="--", linewidth=1.4,
                   label=f"μ={df[col].mean():.1f}")
        ax.set_title(col.replace("_"," ").replace("avg","Avg"), fontsize=9, fontweight="bold")
        ax.legend(fontsize=7.5); ax.set_facecolor("#F9FBE7")
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML MODELS & METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='sec-hdr'>🤖 Model Performance — Random Forest</div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    metric_cards = [
        ("📉 MAE",      reg_metrics["MAE"],          "Regression"),
        ("📐 RMSE",     reg_metrics["RMSE"],          "Regression"),
        ("📊 R² Score", reg_metrics["R2"],            "Regression"),
        ("🎯 Accuracy", f"{cls_metrics['Accuracy']}%","Classification"),
        ("⚡ AUC-ROC",  cls_metrics["AUC_ROC"],       "Classification"),
    ]
    for col, (lbl, val, kind) in zip([m1,m2,m3,m4,m5], metric_cards):
        col.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-val'>{val}</div>
          <div class='kpi-lbl'>{lbl}<br><small>{kind}</small></div>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec-hdr'>🎯 Actual vs Predicted Cases</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="white")
        ax.scatter(yr_te, yr_pred, alpha=0.35, color=GREEN, s=18, edgecolors="none")
        mn, mx = min(yr_te.min(), yr_pred.min()), max(yr_te.max(), yr_pred.max())
        ax.plot([mn,mx],[mn,mx],"r--", linewidth=2, label="Perfect Fit")
        ax.set_xlabel("Actual Cases"); ax.set_ylabel("Predicted Cases")
        ax.set_title("Actual vs Predicted (Regression)", fontweight="bold")
        ax.legend(); ax.set_facecolor("#F9FBE7"); plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown("<div class='sec-hdr'>🌡️ Confusion Matrix (High-Risk Classification)</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4.5), facecolor="white")
        cm = cls_metrics["CM"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
                    xticklabels=["Low Risk","High Risk"], yticklabels=["Low Risk","High Risk"],
                    linewidths=0.5, annot_kws={"size":14})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='sec-hdr'>📈 ROC Curve</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="white")
        ax.plot(cls_metrics["fpr"], cls_metrics["tpr"], color=GREEN, linewidth=2.5,
                label=f"AUC = {cls_metrics['AUC_ROC']:.4f}")
        ax.plot([0,1],[0,1],"--", color="gray", linewidth=1.5, label="Random Guess")
        ax.fill_between(cls_metrics["fpr"], cls_metrics["tpr"], alpha=0.12, color=GREEN)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — High-Risk Classification", fontweight="bold")
        ax.legend(loc="lower right"); ax.set_facecolor("#F9FBE7")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c4:
        st.markdown("<div class='sec-hdr'>🔑 Feature Importances</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="white")
        colors_fi = [RED if v == feat_imp.max() else GREEN for v in feat_imp.values]
        ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors_fi[::-1], edgecolor="white", alpha=0.88)
        ax.set_xlabel("Importance"); ax.set_title("Feature Importances (RF Regressor)", fontweight="bold")
        ax.set_facecolor("#F9FBE7"); plt.tight_layout(); st.pyplot(fig); plt.close()

    # Classification report
    st.markdown("<div class='sec-hdr'>📋 Full Classification Report</div>", unsafe_allow_html=True)
    cls_rep = {
        "Metric": ["Precision","Recall (Sensitivity)","F1 Score","AUC-ROC","Accuracy"],
        "High Risk": [
            cls_metrics["Precision"],
            cls_metrics["Recall"],
            cls_metrics["F1"],
            cls_metrics["AUC_ROC"],
            f"{cls_metrics['Accuracy']}%"
        ],
        "Description": [
            "Proportion of predicted outbreaks that are real (PPV)",
            "Proportion of actual outbreaks correctly identified",
            "Harmonic mean of Precision & Recall",
            "Model ability to discriminate High vs Low Risk",
            "Overall prediction accuracy"
        ]
    }
    st.dataframe(pd.DataFrame(cls_rep), use_container_width=True, hide_index=True)

    # Residuals
    st.markdown("<div class='sec-hdr'>📉 Residuals Distribution</div>", unsafe_allow_html=True)
    residuals = np.array(yr_te) - np.array(yr_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor="white")
    axes[0].hist(residuals, bins=35, color=GREEN, alpha=0.82, edgecolor="white")
    axes[0].axvline(0, color=RED, linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual (Actual − Predicted)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Residuals Histogram", fontweight="bold")
    axes[0].set_facecolor("#F9FBE7")
    axes[1].scatter(yr_pred, residuals, alpha=0.3, color=GREEN, s=12)
    axes[1].axhline(0, color=RED, linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted Cases"); axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted", fontweight="bold")
    axes[1].set_facecolor("#F9FBE7")
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT HIGH-RISK PERIOD
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='sec-hdr'>🔮 Predict Malaria Cases & Risk Level</div>", unsafe_allow_html=True)
    st.info("Enter environmental conditions below to predict expected malaria cases and whether the period is HIGH or LOW risk.")

    col1, col2, col3 = st.columns(3)
    with col1:
        p_rain     = st.slider("🌧️ Rainfall (mm)",          0.0,  250.0, float(df["Rainfall_mm"].mean()), 0.5)
        p_temp     = st.slider("🌡️ Temperature (°C)",       15.0, 40.0,  float(df["Temperature_avg"].mean()), 0.1)
    with col2:
        p_humid    = st.slider("💧 Humidity (%)",            30.0, 100.0, float(df["Humidity_percent"].mean()), 0.5)
        p_mosq     = st.slider("🦟 Mosquito Density",        0.0,  80.0,  float(df["Mosquito_density"].mean()), 0.5)
    with col3:
        p_interv   = st.slider("💉 Intervention Coverage (%)", 0.0, 100.0, float(df["Intervention_coverage"].mean()), 0.5)
        p_month    = st.selectbox("📅 Month", options=list(range(1,13)),
                                  format_func=lambda x: month_names[x-1], index=3)
    p_quarter  = (p_month - 1) // 3 + 1

    if st.button("🔮 Predict Malaria Risk"):
        input_df = pd.DataFrame([{
            "Rainfall_mm":          p_rain,
            "Temperature_avg":      p_temp,
            "Humidity_percent":     p_humid,
            "Mosquito_density":     p_mosq,
            "Intervention_coverage":p_interv,
            "Rainfall_mm_lag1":     p_rain * 0.9,
            "Mosquito_density_lag1":p_mosq * 0.95,
            "Month":                p_month,
            "Quarter":              p_quarter,
        }])

        pred_cases  = rf_reg.predict(input_df)[0]
        pred_risk   = rf_cls.predict(input_df)[0]
        pred_proba  = rf_cls.predict_proba(input_df)[0][1]
        season_name = ("Long Rains (Mar–May)"   if p_month in [3,4,5]   else
                       "Short Rains (Oct–Dec)"  if p_month in [10,11,12] else
                       "Dry Season (Jun–Sep)"   if p_month in [6,7,8,9]  else
                       "Dry Season (Jan–Feb)")

        if pred_risk == 1:
            st.markdown(f"""
            <div class='risk-high'>
              <h3 style='color:{RED};margin:0;'>🚨 HIGH RISK PERIOD</h3>
              <p style='margin:0.4rem 0 0;font-size:1rem;'>
                Predicted Cases: <b>{pred_cases:.0f}</b> &nbsp;|&nbsp;
                Risk Probability: <b>{pred_proba*100:.1f}%</b> &nbsp;|&nbsp;
                Season: <b>{season_name}</b>
              </p>
              <p style='margin:0.5rem 0 0;font-size:0.88rem;color:{RED};'>
                ⚠️ Recommend immediate deployment of ITNs, IRS, and antimalarial drugs.
                Activate early warning protocols.
              </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='risk-low'>
              <h3 style='color:{GREEN};margin:0;'>✅ LOW RISK PERIOD</h3>
              <p style='margin:0.4rem 0 0;font-size:1rem;'>
                Predicted Cases: <b>{pred_cases:.0f}</b> &nbsp;|&nbsp;
                Risk Probability: <b>{pred_proba*100:.1f}%</b> &nbsp;|&nbsp;
                Season: <b>{season_name}</b>
              </p>
              <p style='margin:0.5rem 0 0;font-size:0.88rem;color:{GREEN};'>
                ✔ Maintain standard surveillance. Prepare preventive resources ahead of next rainy season.
              </p>
            </div>""", unsafe_allow_html=True)

        # Risk probability gauge
        st.markdown("**Risk Probability Gauge**")
        fig, ax = plt.subplots(figsize=(8, 1.5), facecolor="white")
        ax.barh(["Risk"], [100], color="#E0E0E0", height=0.5)
        ax.barh(["Risk"], [pred_proba*100],
                color=RED if pred_risk==1 else GREEN, height=0.5)
        ax.axvline(50, color="orange", linestyle="--", linewidth=1.5, label="50% threshold")
        ax.set_xlim(0,100); ax.set_xlabel("Risk Probability (%)")
        ax.set_title(f"Predicted Risk: {pred_proba*100:.1f}%", fontweight="bold")
        ax.legend(fontsize=8); ax.set_facecolor("white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Show nearest historical comparisons
        st.markdown("<div class='sec-hdr'>📋 Similar Historical Records</div>", unsafe_allow_html=True)
        similar = df[df["Month"]==p_month].nsmallest(5, "Total_cases")[
            ["Date","Region","County","Total_cases","Season","Rainfall_mm","Mosquito_density","Risk_Label"]
        ]
        st.dataframe(similar.reset_index(drop=True), use_container_width=True)

    # Batch scenario analysis
    st.markdown("---")
    st.markdown("<div class='sec-hdr'>📊 Scenario Comparison: High vs Low Rainfall Months</div>", unsafe_allow_html=True)
    scenarios = []
    for rain_val, label in [(20,"Low Rain (20mm)"), (80,"Moderate Rain (80mm)"), (150,"High Rain (150mm)"), (220,"Very High Rain (220mm)")]:
        for month_val in range(1,13):
            inp = pd.DataFrame([{
                "Rainfall_mm": rain_val, "Temperature_avg": 26, "Humidity_percent": 72,
                "Mosquito_density": rain_val*0.2, "Intervention_coverage": 65,
                "Rainfall_mm_lag1": rain_val*0.9, "Mosquito_density_lag1": rain_val*0.18,
                "Month": month_val, "Quarter": (month_val-1)//3+1
            }])
            pred = rf_reg.predict(inp)[0]
            risk = rf_cls.predict_proba(inp)[0][1]
            scenarios.append({"Rainfall Scenario": label, "Month": month_names[month_val-1],
                               "Month_Num": month_val, "Predicted Cases": round(pred,1), "Risk Prob %": round(risk*100,1)})
    sc_df = pd.DataFrame(scenarios)
    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor="white")
    palette_sc = {"Low Rain (20mm)":BLUE,"Moderate Rain (80mm)":GOLD,"High Rain (150mm)":ORANGE,"Very High Rain (220mm)":RED}
    for scenario, grp in sc_df.groupby("Rainfall Scenario"):
        ax.plot(grp["Month"], grp["Predicted Cases"], "o-", label=scenario,
                color=palette_sc[scenario], linewidth=2.2, markersize=6)
    ax.set_xticks(range(13)); ax.set_xticklabels([""] + month_names)
    ax.set_ylabel("Predicted Cases"); ax.set_title("Predicted Cases Under Different Rainfall Scenarios", fontweight="bold")
    ax.legend(loc="upper right"); ax.set_facecolor("#F9FBE7"); plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div class='sec-hdr'>🗂️ Dataset Explorer</div>", unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    regions   = fc1.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    counties  = fc2.multiselect("County", sorted(df["County"].unique()), default=sorted(df["County"].unique()))
    seasons   = fc3.multiselect("Season", df["Season"].unique(), default=list(df["Season"].unique()))
    risk_filt = fc4.multiselect("Risk Label", ["High Risk","Low Risk"], default=["High Risk","Low Risk"])

    date_range = st.date_input("Date Range", value=(df["Date"].min(), df["Date"].max()),
                               min_value=df["Date"].min(), max_value=df["Date"].max())

    filtered = df[
        df["Region"].isin(regions) &
        df["County"].isin(counties) &
        df["Season"].isin(seasons) &
        df["Risk_Label"].isin(risk_filt) &
        (df["Date"] >= pd.Timestamp(date_range[0])) &
        (df["Date"] <= pd.Timestamp(date_range[1]))
    ]

    st.success(f"Showing **{len(filtered):,}** records | Total Cases: **{filtered['Total_cases'].sum():,}** | Deaths: **{filtered['Deaths'].sum():,}**")

    display_cols = ["Date","Region","County","Season","Total_cases","Severe_cases",
                    "Deaths","Rainfall_mm","Temperature_avg","Humidity_percent",
                    "Mosquito_density","Intervention_coverage","Risk_Label","CFR"]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=420)

    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="malaria_filtered.csv",
        mime="text/csv"
    )
