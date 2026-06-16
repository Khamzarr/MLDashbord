"""
FinML Studio — Financial Analysis, DCF & Monte Carlo Platform
=============================================================
Профессиональное приложение для:
  • Финансового анализа компаний (LTM, CAGR, margins)
  • DCF-оценки с Monte Carlo симуляцией (10 000 путей)
  • ML-прогнозирования выручки (LR / RF + TimeSeriesSplit)
  • Сценарного анализа с Tornado Chart и Sensitivity Table

Стек: Python 3.11 | Streamlit | Pandas | NumPy | Plotly | Scikit-Learn | OpenPyXL
"""

import io
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# КОНСТАНТЫ
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE    = "FinML Studio"
APP_SUBTITLE = "DCF · Monte Carlo · ML Forecasting · Risk Analysis"

C = dict(
    blue   = "#4f8ef7",
    green  = "#22c55e",
    red    = "#ef4444",
    amber  = "#f59e0b",
    purple = "#a855f7",
    teal   = "#14b8a6",
    pink   = "#f472b6",
    indigo = "#6366f1",
)

TPL = "plotly_dark"
LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#8b95aa"),
    margin=dict(l=12, r=12, t=44, b=12),
    xaxis=dict(gridcolor="#1e2535", color="#8b95aa", zerolinecolor="#1e2535"),
    yaxis=dict(gridcolor="#1e2535", color="#8b95aa", zerolinecolor="#1e2535"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11)),
    hovermode="x unified",
)

REQUIRED_COLS = [
    "Date","Revenue","COGS","EBITDA",
    "Net_Income","CAPEX","Working_Capital","Free_Cash_Flow",
]

# ─────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE, page_icon="📊",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:#0a0e1a;}
/* header */
.fin-header{background:linear-gradient(135deg,#111827 0%,#1a2035 100%);
  border:1px solid #1e2d45;border-radius:14px;padding:22px 28px;margin-bottom:20px;}
.fin-title{font-size:28px;font-weight:800;
  background:linear-gradient(90deg,#4f8ef7 0%,#a855f7 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;}
.fin-sub{color:#6b7a99;font-size:13px;margin:3px 0 0;}
/* metrics */
div[data-testid="metric-container"]{
  background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:14px 16px;}
div[data-testid="metric-container"] label{color:#6b7a99!important;font-size:11px!important;font-weight:500!important;text-transform:uppercase;letter-spacing:.04em;}
div[data-testid="metric-container"] div[data-testid="metric-value"]{color:#e2e8f0!important;font-size:21px!important;font-weight:700!important;}
/* cards */
.card{background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:16px 18px;margin-bottom:14px;}
.card-title{font-size:13px;font-weight:600;color:#e2e8f0;margin-bottom:10px;}
/* info */
.info-box{background:rgba(79,142,247,.07);border:1px solid rgba(79,142,247,.2);
  border-radius:8px;padding:9px 13px;font-size:12px;color:#8b95aa;margin-bottom:12px;}
.warn-box{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2);
  border-radius:8px;padding:9px 13px;font-size:12px;color:#a38050;margin-bottom:12px;}
/* sidebar */
section[data-testid="stSidebar"]{background:#0d1320;border-right:1px solid #1e2d45;}
section[data-testid="stSidebar"] label{color:#6b7a99!important;font-size:11px!important;font-weight:500!important;}
/* tabs */
.stTabs [data-baseweb="tab-list"]{background:#111827;border-radius:9px;padding:3px;gap:3px;}
.stTabs [data-baseweb="tab"]{border-radius:6px;color:#6b7a99;font-weight:500;font-size:12px;}
.stTabs [aria-selected="true"]{background:#1e2d45!important;color:#e2e8f0!important;}
/* dataframe */
.stDataFrame{border-radius:8px;overflow:hidden;}
/* hide streamlit chrome */
#MainMenu,footer,header{visibility:hidden;}
div[data-testid="stToolbar"]{display:none;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# СИНТЕТИЧЕСКИЕ ДАННЫЕ
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def generate_synthetic(seed: int = 42) -> pd.DataFrame:
    """
    Генерирует реалистичные финансовые данные за 5 лет (20 кварталов).
    Включает тренд роста, сезонность Q1-Q4 и случайный шум.
    """
    np.random.seed(seed)
    n = 20
    dates = pd.date_range("2020-01-01", periods=n, freq="QS")

    # Тренд: 12% CAGR → ~2.9% в квартал
    qg    = (1.12 ** 0.25) - 1
    trend = np.array([(1 + qg) ** i for i in range(n)])
    seas  = np.tile([0.84, 0.96, 1.06, 1.14], 5)
    noise = 1 + np.random.normal(0, 0.045, n)

    rev  = 500 * trend * seas * noise

    cogs_r  = np.clip(np.linspace(0.60, 0.55, n) + np.random.normal(0, .01, n), .50, .66)
    cogs    = rev * cogs_r

    ebitda_r = np.clip(np.linspace(0.20, 0.28, n) + np.random.normal(0, .013, n), .14, .36)
    ebitda   = rev * ebitda_r

    da      = rev * 0.05
    ebit    = ebitda - da
    ni      = np.maximum(ebit * 0.80 * (1 + np.random.normal(0, .025, n)), rev * .01)

    capex_r = np.clip(np.linspace(0.09, 0.07, n) + np.random.normal(0, .008, n), .04, .13)
    capex   = rev * capex_r

    wc      = rev * (0.15 + np.linspace(0, .03, n))
    dwc     = np.diff(wc, prepend=wc[0] * 0.98)
    fcf     = ni + da - capex - dwc

    return pd.DataFrame({
        "Date":            dates,
        "Revenue":         np.round(rev,  2),
        "COGS":            np.round(cogs, 2),
        "EBITDA":          np.round(ebitda, 2),
        "Net_Income":      np.round(ni,   2),
        "CAPEX":           np.round(capex,2),
        "Working_Capital": np.round(wc,   2),
        "Free_Cash_Flow":  np.round(fcf,  2),
    })


# ─────────────────────────────────────────────────────────────────────────────
# ЗАГРУЗКА ФАЙЛА
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_file(f) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Загружает CSV или XLSX, валидирует колонки."""
    try:
        name = f.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(f)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f, engine="openpyxl")
        else:
            return None, "Поддерживаются CSV и XLSX."

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            return None, f"Отсутствуют колонки: {', '.join(missing)}"

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        for c in REQUIRED_COLS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# ФИНАНСОВЫЕ КОЭФФИЦИЕНТЫ
# ─────────────────────────────────────────────────────────────────────────────

def calc_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает финансовые коэффициенты на основе LTM данных."""
    ltm = df.iloc[-4:] if len(df) >= 4 else df
    rev   = ltm["Revenue"].sum()
    ebitda= ltm["EBITDA"].sum()
    ni    = ltm["Net_Income"].sum()
    capex = ltm["CAPEX"].sum()
    cogs  = ltm["COGS"].sum()
    fcf   = ltm["Free_Cash_Flow"].sum()

    gm   = (rev - cogs) / rev * 100 if rev else 0
    em   = ebitda / rev * 100 if rev else 0
    nm   = ni / rev * 100 if rev else 0
    cr   = capex / rev * 100 if rev else 0
    fcfm = fcf / rev * 100 if rev else 0

    n_yr = len(df) / 4
    cagr = ((df["Revenue"].iloc[-1] / df["Revenue"].iloc[0]) ** (1/n_yr) - 1) * 100 \
           if n_yr > 0 and df["Revenue"].iloc[0] > 0 else 0

    ic   = rev * 1.5  # proxy invested capital
    roic = ni / ic * 100 if ic > 0 else 0

    def grade(v, hi, lo, inv=False):
        if inv:
            return "✅ Отлично" if v < lo else ("⚠️ Норма" if v < hi else "❌ Высокий")
        return "✅ Отлично" if v > hi else ("⚠️ Норма" if v > lo else "❌ Низко")

    return pd.DataFrame({
        "Показатель": ["Gross Margin","EBITDA Margin","Net Margin",
                       "FCF Margin","Revenue CAGR","ROIC","CAPEX / Revenue"],
        "Значение":   [f"{gm:.1f}%", f"{em:.1f}%", f"{nm:.1f}%",
                       f"{fcfm:.1f}%", f"{cagr:.1f}%", f"{roic:.1f}%", f"{cr:.1f}%"],
        "Оценка":     [grade(gm,40,25), grade(em,25,15), grade(nm,15,8),
                       grade(fcfm,12,5), grade(cagr,15,5), grade(roic,15,8),
                       grade(cr,7,12,inv=True)],
    })


# ─────────────────────────────────────────────────────────────────────────────
# DCF — ДЕТЕРМИНИРОВАННЫЙ
# ─────────────────────────────────────────────────────────────────────────────

def run_dcf(
    df: pd.DataFrame,
    wacc: float, tgr: float, rev_g: float,
    ebitda_m: float, horizon: int,
    da_r: float = 0.05, capex_r: float = 0.07,
    tax: float = 0.20, nwc_r: float = 0.02,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Детерминированная DCF-модель.
    Возвращает таблицу прогноза и словарь с итоговыми оценками.
    """
    base = df.iloc[-4:]["Revenue"].sum() if len(df) >= 4 else df["Revenue"].sum()
    tgr  = min(tgr, wacc - 0.01)  # защита от деления на 0

    rows = []
    for t in range(1, horizon + 1):
        rev   = base * (1 + rev_g) ** t
        ebitda= rev * ebitda_m
        da    = rev * da_r
        ebit  = ebitda - da
        nopat = max(ebit * (1 - tax), 0)
        capex = rev * capex_r
        dnwc  = rev * nwc_r
        fcf   = nopat + da - capex - dnwc
        df_   = 1 / (1 + wacc) ** t
        rows.append(dict(
            Год=f"Г{t}", Выручка=rev, EBITDA=ebitda, EBIT=ebit,
            NOPAT=nopat, DA=da, CAPEX=capex, ΔNWC=dnwc,
            FCF=fcf, Дисконт=df_, PV_FCF=fcf*df_,
        ))

    tbl = pd.DataFrame(rows)
    tbl = tbl.round(2)

    last_fcf  = tbl["FCF"].iloc[-1]
    tv        = last_fcf * (1 + tgr) / (wacc - tgr)
    pv_tv     = tv / (1 + wacc) ** horizon
    sum_pv    = tbl["PV_FCF"].sum()
    ev        = sum_pv + pv_tv
    net_debt  = base * 0.30
    eq        = ev - net_debt

    summary = dict(
        sum_pv=round(sum_pv,2), tv=round(tv,2), pv_tv=round(pv_tv,2),
        ev=round(ev,2), net_debt=round(net_debt,2), eq=round(eq,2),
        base=round(base,2), ev_rev=round(ev/base,2) if base else 0,
    )
    return tbl, summary


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO DCF
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def monte_carlo_dcf(
    base_rev: float,
    wacc_mu: float,    wacc_sigma: float,
    rev_g_mu: float,   rev_g_sigma: float,
    em_mu: float,      em_sigma: float,
    tgr_mu: float,     tgr_sigma: float,
    horizon: int,
    n_sim: int = 10_000,
    da_r: float = 0.05,
    capex_r: float = 0.07,
    tax: float = 0.20,
    nwc_r: float = 0.02,
    seed: int = 0,
) -> Dict:
    """
    Monte Carlo симуляция DCF: n_sim независимых сценариев.

    Для каждого сценария семплируем:
      • WACC          ~ N(wacc_mu, wacc_sigma),  clip [2%, 35%]
      • Revenue Growth~ N(rev_g_mu, rev_g_sigma), clip [-30%, 50%]
      • EBITDA Margin ~ N(em_mu, em_sigma),       clip [3%, 60%]
      • Terminal GR   ~ N(tgr_mu, tgr_sigma),     clip [0%, 5%]

    Возвращает массивы EV, Equity Value и ключевую статистику.
    """
    rng = np.random.default_rng(seed)

    waccs   = np.clip(rng.normal(wacc_mu,  wacc_sigma,  n_sim), 0.02, 0.35)
    rev_gs  = np.clip(rng.normal(rev_g_mu, rev_g_sigma, n_sim), -0.30, 0.50)
    ems     = np.clip(rng.normal(em_mu,    em_sigma,    n_sim), 0.03, 0.60)
    tgrs    = np.clip(rng.normal(tgr_mu,   tgr_sigma,   n_sim), 0.00, 0.05)
    tgrs    = np.minimum(tgrs, waccs - 0.005)

    ev_arr  = np.zeros(n_sim)
    fcf_yr1 = np.zeros(n_sim)

    for i in range(n_sim):
        w, g, m, tg = waccs[i], rev_gs[i], ems[i], tgrs[i]
        sum_pv = 0.0
        last_fcf = 0.0
        for t in range(1, horizon + 1):
            rev   = base_rev * (1 + g) ** t
            ebitda= rev * m
            da    = rev * da_r
            ebit  = ebitda - da
            nopat = max(ebit * (1 - tax), 0.0)
            fcf   = nopat + da - rev * capex_r - rev * nwc_r
            pv    = fcf / (1 + w) ** t
            sum_pv += pv
            last_fcf = fcf
            if t == 1:
                fcf_yr1[i] = fcf
        tv      = last_fcf * (1 + tg) / (w - tg) if (w - tg) > 0.001 else 0
        pv_tv   = tv / (1 + w) ** horizon
        ev_arr[i] = sum_pv + pv_tv

    net_debt  = base_rev * 0.30
    eq_arr    = ev_arr - net_debt

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    ev_pct  = {p: float(np.percentile(ev_arr, p))  for p in percentiles}
    eq_pct  = {p: float(np.percentile(eq_arr, p))  for p in percentiles}

    return dict(
        ev_arr=ev_arr, eq_arr=eq_arr,
        ev_pct=ev_pct, eq_pct=eq_pct,
        ev_mean=float(ev_arr.mean()),  ev_std=float(ev_arr.std()),
        eq_mean=float(eq_arr.mean()),  eq_std=float(eq_arr.std()),
        prob_positive_eq=float((eq_arr > 0).mean()),
        prob_ev_above_base=float((ev_arr > base_rev).mean()),
        waccs=waccs, rev_gs=rev_gs, ems=ems, tgrs=tgrs,
        fcf_yr1=fcf_yr1, net_debt=net_debt,
        n_sim=n_sim,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ML — ПРОГНОЗ ВЫРУЧКИ
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def train_ml(df: pd.DataFrame, model_type: str) -> Tuple:
    """
    Обучает модель прогноза Revenue с TimeSeriesSplit кросс-валидацией.
    Признаки: тренд, квартал, год, лаги 1/2/4, MA4, YoY growth.
    """
    d = df.copy().reset_index(drop=True)
    d["t"]       = np.arange(len(d))
    d["quarter"] = d["Date"].dt.quarter
    d["year"]    = d["Date"].dt.year
    d["lag1"]    = d["Revenue"].shift(1)
    d["lag2"]    = d["Revenue"].shift(2)
    d["lag4"]    = d["Revenue"].shift(4)
    d["ma4"]     = d["Revenue"].rolling(4).mean()
    d["yoy"]     = d["Revenue"] / d["Revenue"].shift(4) - 1
    d = d.dropna().reset_index(drop=True)

    FEATS = ["t","quarter","year","lag1","lag2","lag4","ma4","yoy"]
    X, y  = d[FEATS].values, d["Revenue"].values

    # TimeSeriesSplit кросс-валидация
    tscv = TimeSeriesSplit(n_splits=3)
    mae_scores, rmse_scores, r2_scores = [], [], []

    scaler = StandardScaler()
    if model_type == "Linear Regression":
        mdl = LinearRegression()
    else:
        mdl = RandomForestRegressor(n_estimators=300, max_depth=8,
                                    min_samples_leaf=2, random_state=42)

    for tr_idx, te_idx in tscv.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        mdl.fit(Xtr_s, ytr)
        yp = mdl.predict(Xte_s)
        mae_scores.append(mean_absolute_error(yte, yp))
        rmse_scores.append(np.sqrt(mean_squared_error(yte, yp)))
        r2_scores.append(r2_score(yte, yp))

    # Финальное обучение на всех данных
    scaler.fit(X)
    mdl.fit(scaler.transform(X), y)

    metrics = dict(
        MAE  = round(np.mean(mae_scores), 2),
        RMSE = round(np.mean(rmse_scores), 2),
        R2   = round(np.mean(r2_scores), 4),
    )
    return mdl, scaler, d, FEATS, metrics


def forecast_ml(mdl, scaler, d, feats, n=12) -> pd.DataFrame:
    """Строит прогноз на n кварталов вперёд (рекурсивно)."""
    last_t    = int(d["t"].max())
    last_date = d["Date"].max()
    buf       = list(d["Revenue"].values)
    rows = []

    for i in range(1, n + 1):
        cur_date = last_date + pd.DateOffset(months=3 * i)
        t_    = last_t + i
        q_    = ((cur_date.month - 1) // 3) + 1
        yr_   = cur_date.year
        lag1  = buf[-1]
        lag2  = buf[-2] if len(buf) >= 2 else lag1
        lag4  = buf[-4] if len(buf) >= 4 else lag1
        ma4   = np.mean(buf[-4:]) if len(buf) >= 4 else lag1
        yoy   = (lag1 / buf[-4] - 1) if len(buf) >= 4 and buf[-4] > 0 else 0
        X_new = np.array([[t_, q_, yr_, lag1, lag2, lag4, ma4, yoy]])
        pred  = float(mdl.predict(scaler.transform(X_new))[0])
        buf.append(pred)
        rows.append({"Date": cur_date, "Revenue": round(pred, 2)})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ЭКСПОРТ
# ─────────────────────────────────────────────────────────────────────────────

def to_xlsx(dfs: Dict[str, pd.DataFrame]) -> bytes:
    """Экспортирует несколько DataFrame в один XLSX файл."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for sh, df in dfs.items():
            df.to_excel(w, sheet_name=sh[:31], index=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v: float) -> str:
    """Форматирует число в читаемый вид (млн/млрд)."""
    if abs(v) >= 1_000:
        return f"{v/1000:,.1f} млрд"
    return f"{v:,.1f} млн"


def apply_layout(fig: go.Figure, title: str = "", h: int = 320) -> go.Figure:
    """Применяет стандартный стиль к фигуре Plotly."""
    kw = {**LAYOUT_BASE, "height": h, "template": TPL}
    if title:
        kw["title"] = dict(text=title, font=dict(size=13, color="#c4cfe0"))
    fig.update_layout(**kw)
    return fig


def line_fig(df, x, ys, title="", h=300, colors=None) -> go.Figure:
    """Линейный график для нескольких серий."""
    fig  = go.Figure()
    pal  = [C["blue"], C["green"], C["amber"], C["purple"], C["teal"]]
    cols = colors or pal
    for i, y in enumerate(ys):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y], name=y,
            line=dict(color=cols[i % len(cols)], width=2.2),
            mode="lines+markers", marker=dict(size=3.5),
        ))
    return apply_layout(fig, title, h)


def bar_fig(df, x, ys, title="", h=300, barmode="group") -> go.Figure:
    """Барчарт для нескольких серий."""
    fig = go.Figure()
    pal = [C["blue"], C["green"], C["amber"], C["purple"]]
    for i, y in enumerate(ys):
        fig.add_trace(go.Bar(
            x=df[x], y=df[y], name=y,
            marker_color=pal[i % len(pal)],
        ))
    fig.update_layout(barmode=barmode)
    return apply_layout(fig, title, h)


# ─────────────────────────────────────────────────────────────────────────────
# САЙДБАР
# ─────────────────────────────────────────────────────────────────────────────

def sidebar() -> Dict:
    """Рендерит сайдбар и возвращает параметры."""
    with st.sidebar:
        st.markdown("## ⚙️ Параметры")
        st.markdown("---")

        st.markdown("### 📂 Данные")
        up = st.file_uploader(
            "CSV / XLSX",
            type=["csv","xlsx","xls"],
            help="Колонки: Date, Revenue, COGS, EBITDA, Net_Income, CAPEX, Working_Capital, Free_Cash_Flow",
        )

        st.markdown("---")
        st.markdown("### 💰 DCF Base Case")
        wacc     = st.slider("WACC (%)",             5,  25, 12) / 100
        tgr      = st.slider("Terminal Growth (%)",  1,   5,  2) / 100
        rev_g    = st.slider("Rev Growth (%)",      -10, 25, 10) / 100
        ebitda_m = st.slider("EBITDA Margin (%)",    5,  40, 25) / 100
        horizon  = st.slider("Горизонт (лет)",        3,  10,  5)

        st.markdown("---")
        st.markdown("### 🎲 Monte Carlo")
        n_sim       = st.select_slider("Симуляций", [1_000, 5_000, 10_000, 50_000], 10_000)
        wacc_sigma  = st.slider("σ WACC (%)",          0.5,  5.0, 1.5) / 100
        rev_sigma   = st.slider("σ Rev Growth (%)",    1.0, 10.0, 3.0) / 100
        em_sigma    = st.slider("σ EBITDA Margin (%)", 1.0,  8.0, 2.5) / 100
        tgr_sigma   = st.slider("σ Terminal GR (%)",   0.1,  2.0, 0.5) / 100

        st.markdown("---")
        st.markdown("### 🤖 Machine Learning")
        mdl_type = st.selectbox("Модель", ["Random Forest Regressor","Linear Regression"])
        n_fore   = st.slider("Периодов прогноза", 4, 20, 12)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:10px;color:#3a4a66;text-align:center;'>"
            "FinML Studio v3.0 · Monte Carlo Edition</div>",
            unsafe_allow_html=True,
        )

    return dict(
        uploaded=up, wacc=wacc, tgr=tgr, rev_g=rev_g,
        ebitda_m=ebitda_m, horizon=horizon,
        n_sim=n_sim, wacc_sigma=wacc_sigma, rev_sigma=rev_sigma,
        em_sigma=em_sigma, tgr_sigma=tgr_sigma,
        mdl_type=mdl_type, n_fore=n_fore,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def tab_dashboard(df: pd.DataFrame) -> None:
    """KPI метрики и основные тренды."""
    ltm  = df.iloc[-4:] if len(df) >= 4 else df
    prev = df.iloc[-8:-4] if len(df) >= 8 else df.iloc[:max(1, len(df)-4)]

    rev_l  = ltm["Revenue"].sum();     rev_p  = prev["Revenue"].sum()
    ebi_l  = ltm["EBITDA"].sum();      ni_l   = ltm["Net_Income"].sum()
    fcf_l  = ltm["Free_Cash_Flow"].sum()
    rev_d  = (rev_l/rev_p - 1)*100 if rev_p > 0 else 0
    ebi_m  = ebi_l/rev_l*100 if rev_l else 0
    ni_m   = ni_l/rev_l*100  if rev_l else 0
    fcf_m  = fcf_l/rev_l*100 if rev_l else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("💰 Revenue LTM",    fmt(rev_l),  f"{rev_d:+.1f}% г/г")
    c2.metric("📈 EBITDA LTM",     fmt(ebi_l),  f"Маржа {ebi_m:.1f}%")
    c3.metric("💵 Net Income LTM", fmt(ni_l),   f"Маржа {ni_m:.1f}%")
    c4.metric("🌊 FCF LTM",        fmt(fcf_l),  f"Yield {fcf_m:.1f}%")
    c5.metric("🚀 Rev Growth г/г", f"{rev_d:+.1f}%", "LTM vs PY")
    c6.metric("📅 Периодов",       str(len(df)), "кварталов")

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(
            line_fig(df, "Date", ["Revenue","EBITDA"], "Revenue & EBITDA", 290),
            use_container_width=True,
        )
    with cr:
        st.plotly_chart(
            line_fig(df, "Date", ["Net_Income","Free_Cash_Flow"],
                     "Net Income & FCF", 290,
                     colors=[C["green"], C["teal"]]),
            use_container_width=True,
        )

    # Годовой бар
    df2 = df.copy()
    df2["Year"] = df2["Date"].dt.year
    ann = df2.groupby("Year")[["Revenue","EBITDA","Net_Income"]].sum().reset_index()
    ann["Year"] = ann["Year"].astype(str)
    st.plotly_chart(
        bar_fig(ann, "Year", ["Revenue","EBITDA","Net_Income"],
                "Годовые показатели", 270),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────

def tab_eda(df: pd.DataFrame) -> None:
    """Exploratory Data Analysis: таблица, статистика, корреляции."""
    st.markdown('<div class="info-box">Выберите период для анализа.</div>',
                unsafe_allow_html=True)

    periods = [str(p) for p in sorted(df["Date"].dt.to_period("Q").unique())]
    cf, ct = st.columns(2)
    f_idx = cf.selectbox("Начало", periods, 0)
    t_idx = ct.selectbox("Конец",  periods, len(periods)-1)

    fp = pd.Period(f_idx, "Q");  tp = pd.Period(t_idx, "Q")
    mask = (df["Date"].dt.to_period("Q") >= fp) & (df["Date"].dt.to_period("Q") <= tp)
    dv   = df[mask].copy()

    if dv.empty:
        st.warning("Нет данных."); return

    st.markdown("#### 📋 Данные")
    num_cols = REQUIRED_COLS[1:]
    st.dataframe(
        dv.style.format({c: "{:,.1f}" for c in num_cols if c in dv}),
        use_container_width=True, height=260,
    )

    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### 📐 Описательная статистика")
        st.dataframe(dv[num_cols].describe().round(2), use_container_width=True)

    with cb:
        st.markdown("#### 🏆 Финансовые коэффициенты")
        st.dataframe(calc_ratios(dv), use_container_width=True, hide_index=True)

    st.markdown("#### 🔗 Корреляционная матрица")
    corr = dv[num_cols].corr().round(3)
    fig_c = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f", template=TPL, height=380,
    )
    fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig_c, use_container_width=True)

    cl2, cr2 = st.columns(2)
    for col, container in [("Revenue",cl2),("EBITDA",cr2),
                           ("Net_Income",cl2),("Free_Cash_Flow",cr2)]:
        with container:
            st.plotly_chart(
                line_fig(dv,"Date",[col],col, 230),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DCF VALUATION
# ─────────────────────────────────────────────────────────────────────────────

def tab_dcf(df: pd.DataFrame, p: Dict) -> None:
    """Детерминированный DCF с таблицей, Waterfall и экспортом."""
    st.markdown(
        f'<div class="info-box">WACC={p["wacc"]*100:.0f}% · '
        f'TGR={p["tgr"]*100:.0f}% · RevG={p["rev_g"]*100:.0f}% · '
        f'EBITDA%={p["ebitda_m"]*100:.0f}% · {p["horizon"]} лет</div>',
        unsafe_allow_html=True,
    )

    try:
        tbl, s = run_dcf(df, p["wacc"], p["tgr"], p["rev_g"],
                         p["ebitda_m"], p["horizon"])
    except Exception as e:
        st.error(f"DCF error: {e}"); return

    # KPI
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🏢 Enterprise Value", fmt(s["ev"]))
    c2.metric("💎 Equity Value",     fmt(s["eq"]))
    c3.metric("🌊 PV of FCFs",       fmt(s["sum_pv"]))
    c4.metric("📡 EV / Revenue",     f"{s['ev_rev']:.1f}x")

    cl, cr = st.columns([3,2])
    with cl:
        st.markdown("#### 📋 DCF таблица")
        fmt_d = {c: "{:,.1f}" for c in tbl.columns if c not in ["Год","Дисконт"]}
        fmt_d["Дисконт"] = "{:.4f}"
        st.dataframe(tbl.style.format(fmt_d),
                     use_container_width=True, hide_index=True)

        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(x=tbl["Год"], y=tbl["FCF"],    name="FCF",
                               marker_color=C["blue"]))
        fig_b.add_trace(go.Bar(x=tbl["Год"], y=tbl["PV_FCF"], name="PV FCF",
                               marker_color=C["teal"]))
        apply_layout(fig_b, "FCF vs PV FCF по годам", 280)
        fig_b.update_layout(barmode="group")
        st.plotly_chart(fig_b, use_container_width=True)

    with cr:
        st.markdown("#### 🌊 Waterfall оценки")
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","total","relative","total"],
            x=["PV FCFs","PV Terminal","Enterprise Value","Долг","Equity Value"],
            y=[s["sum_pv"], s["pv_tv"], 0, -s["net_debt"], 0],
            text=[fmt(s["sum_pv"]), fmt(s["pv_tv"]), "",
                  f"−{fmt(s['net_debt'])}", ""],
            textposition="outside",
            decreasing=dict(marker_color=C["red"]),
            increasing=dict(marker_color=C["green"]),
            totals=dict(marker_color=C["blue"]),
            connector=dict(line=dict(color="#1e2d45")),
        ))
        apply_layout(fig_wf, "", 360)
        fig_wf.update_layout(showlegend=False)
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown(f"""
        <div class="card">
          <div class="card-title">🎯 Итог</div>
          <table style="width:100%;font-size:13px;color:#e2e8f0;border-collapse:collapse;">
            <tr><td style="color:#6b7a99;padding:3px 0;">Enterprise Value</td>
                <td style="text-align:right;font-weight:700;">{fmt(s["ev"])}</td></tr>
            <tr><td style="color:#6b7a99;padding:3px 0;">PV Terminal Value</td>
                <td style="text-align:right;">{fmt(s["pv_tv"])}</td></tr>
            <tr><td style="color:#6b7a99;padding:3px 0;">Чистый долг</td>
                <td style="text-align:right;color:{C['red']};">−{fmt(s["net_debt"])}</td></tr>
            <tr style="border-top:1px solid #1e2d45;">
                <td style="color:{C['green']};padding:5px 0;font-weight:700;">Equity Value</td>
                <td style="text-align:right;color:{C['green']};font-weight:700;font-size:15px;">{fmt(s["eq"])}</td></tr>
            <tr><td style="color:#6b7a99;">EV / Revenue</td>
                <td style="text-align:right;font-weight:600;">{s["ev_rev"]:.1f}x</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # Sensitivity table: EV при разных WACC × RevGrowth
    st.markdown("#### 🔢 Sensitivity Table — Enterprise Value (EV / Revenue mult.)")
    waccs_range = [p["wacc"] - 0.03, p["wacc"] - 0.015, p["wacc"],
                   p["wacc"] + 0.015, p["wacc"] + 0.03]
    revg_range  = [p["rev_g"] - 0.04, p["rev_g"] - 0.02, p["rev_g"],
                   p["rev_g"] + 0.02, p["rev_g"] + 0.04]
    sens_rows = []
    for w in waccs_range:
        row = {}
        for g in revg_range:
            try:
                _, ss = run_dcf(df, max(w, 0.03), p["tgr"], g,
                                p["ebitda_m"], p["horizon"])
                row[f"RevG {g*100:+.0f}%"] = f"{ss['ev_rev']:.1f}x"
            except:
                row[f"RevG {g*100:+.0f}%"] = "—"
        row["WACC"] = f"{w*100:.1f}%"
        sens_rows.append(row)

    sens_df = pd.DataFrame(sens_rows).set_index("WACC")
    st.dataframe(
        sens_df.style.map(
            lambda v: f"color: {C['green']}" if "x" in str(v) and float(str(v).replace("x","")) > 5
                      else (f"color: {C['red']}" if "x" in str(v) and float(str(v).replace("x","")) < 2 else "")
        ),
        use_container_width=True,
    )

    # Экспорт
    xlsx_bytes = to_xlsx({"DCF": tbl, "Sensitivity": sens_df.reset_index()})
    st.download_button("📥 Скачать DCF + Sensitivity (XLSX)", xlsx_bytes,
                       "dcf_valuation.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def tab_montecarlo(df: pd.DataFrame, p: Dict) -> None:
    """
    Monte Carlo симуляция DCF.
    Показывает распределение EV, Equity Value,
    VaR/CVaR, доверительные интервалы и корреляцию параметров.
    """
    base_rev = df.iloc[-4:]["Revenue"].sum() if len(df) >= 4 else df["Revenue"].sum()

    st.markdown(
        f'<div class="info-box">🎲 {p["n_sim"]:,} симуляций · '
        f'Base WACC={p["wacc"]*100:.0f}% ±{p["wacc_sigma"]*100:.1f}% · '
        f'Base RevG={p["rev_g"]*100:.0f}% ±{p["rev_sigma"]*100:.1f}% · '
        f'Горизонт {p["horizon"]} лет</div>',
        unsafe_allow_html=True,
    )

    with st.spinner(f"Запуск {p['n_sim']:,} симуляций..."):
        mc = monte_carlo_dcf(
            base_rev     = base_rev,
            wacc_mu      = p["wacc"],      wacc_sigma  = p["wacc_sigma"],
            rev_g_mu     = p["rev_g"],     rev_g_sigma = p["rev_sigma"],
            em_mu        = p["ebitda_m"],  em_sigma    = p["em_sigma"],
            tgr_mu       = p["tgr"],       tgr_sigma   = p["tgr_sigma"],
            horizon      = p["horizon"],   n_sim       = p["n_sim"],
        )

    ev  = mc["ev_arr"]
    eq  = mc["eq_arr"]
    ep  = mc["ev_pct"]
    qp  = mc["eq_pct"]

    # ── KPI ──────────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("📊 EV медиана",     fmt(ep[50]))
    c2.metric("📊 EV среднее",     fmt(mc["ev_mean"]))
    c3.metric("💎 Equity медиана", fmt(qp[50]))
    c4.metric("⬇️ EV VaR 5%",     fmt(ep[5]),
              f"Потери vs медиана: {fmt(ep[5]-ep[50])}")
    c5.metric("⬆️ EV 95%",        fmt(ep[95]))
    c6.metric("✅ P(Equity>0)",    f"{mc['prob_positive_eq']*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Гистограмма EV ──────────────────────────────────────────────────────
    cl, cr = st.columns([3,2])
    with cl:
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Histogram(
            x=ev, nbinsx=120, name="EV",
            marker_color=C["blue"], opacity=0.75,
        ))
        # VaR линии
        for pct, color, label in [
            (5,  C["red"],    "VaR 5%"),
            (25, C["amber"],  "P25"),
            (50, C["green"],  "Медиана"),
            (75, C["teal"],   "P75"),
            (95, C["purple"], "P95"),
        ]:
            fig_ev.add_vline(
                x=ep[pct], line_dash="dash",
                line_color=color, line_width=1.5,
                annotation_text=f"{label}: {fmt(ep[pct])}",
                annotation_font=dict(size=10, color=color),
                annotation_position="top right" if pct >= 50 else "top left",
            )
        apply_layout(fig_ev, f"Распределение Enterprise Value ({p['n_sim']:,} симуляций)", 360)
        fig_ev.update_layout(
            xaxis_title="EV (млн руб.)",
            yaxis_title="Частота",
            showlegend=False,
        )
        st.plotly_chart(fig_ev, use_container_width=True)

    with cr:
        # Таблица перцентилей
        st.markdown("#### 📊 Перцентили EV / Equity")
        pct_labels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_df = pd.DataFrame({
            "Перцентиль": [f"P{p_}" for p_ in pct_labels],
            "EV":         [fmt(ep[p_]) for p_ in pct_labels],
            "Equity":     [fmt(qp[p_]) for p_ in pct_labels],
        })
        st.dataframe(pct_df, use_container_width=True, hide_index=True, height=320)

        # Вероятности
        det_ev = run_dcf(
            df, p["wacc"], p["tgr"], p["rev_g"], p["ebitda_m"], p["horizon"]
        )[1]["ev"]

        p_above = float((ev > det_ev).mean()) * 100
        st.markdown(f"""
        <div class="card" style="margin-top:10px;">
          <div class="card-title">🎯 Вероятностные метрики</div>
          <table style="width:100%;font-size:12px;color:#e2e8f0;border-collapse:collapse;">
            <tr><td style="color:#6b7a99;padding:3px 0;">P(EV > Base Case)</td>
                <td style="text-align:right;font-weight:700;color:{C['green']};">{p_above:.1f}%</td></tr>
            <tr><td style="color:#6b7a99;padding:3px 0;">P(Equity > 0)</td>
                <td style="text-align:right;font-weight:700;color:{C['green']};">{mc['prob_positive_eq']*100:.1f}%</td></tr>
            <tr><td style="color:#6b7a99;padding:3px 0;">EV std (1σ)</td>
                <td style="text-align:right;font-weight:700;">±{fmt(mc['ev_std'])}</td></tr>
            <tr><td style="color:#6b7a99;padding:3px 0;">CV (std/mean)</td>
                <td style="text-align:right;font-weight:700;">{mc['ev_std']/mc['ev_mean']*100:.1f}%</td></tr>
            <tr><td style="color:#6b7a99;padding:3px 0;">Base Case EV</td>
                <td style="text-align:right;font-weight:700;color:{C['blue']};">{fmt(det_ev)}</td></tr>
            <tr><td style="color:#6b7a99;padding:3px 0;">Base vs Медиана</td>
                <td style="text-align:right;font-weight:700;color:{C['amber']};">{fmt(det_ev - ep[50])}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Гистограмма Equity Value ─────────────────────────────────────────────
    st.markdown("#### 💎 Распределение Equity Value")
    fig_eq = go.Figure()
    colors_eq = [C["green"] if v > 0 else C["red"] for v in eq]

    # Упрощённый вариант — гистограмма с делением на зоны
    eq_pos = eq[eq > 0]
    eq_neg = eq[eq <= 0]

    if len(eq_pos):
        fig_eq.add_trace(go.Histogram(
            x=eq_pos, nbinsx=80, name="Equity > 0",
            marker_color=C["green"], opacity=0.7,
        ))
    if len(eq_neg):
        fig_eq.add_trace(go.Histogram(
            x=eq_neg, nbinsx=30, name="Equity ≤ 0",
            marker_color=C["red"], opacity=0.7,
        ))

    fig_eq.add_vline(x=0, line_color="white", line_width=1.5, line_dash="dot")
    fig_eq.add_vline(
        x=qp[50], line_color=C["green"], line_width=2, line_dash="dash",
        annotation_text=f"Медиана: {fmt(qp[50])}",
        annotation_font=dict(size=10, color=C["green"]),
    )
    apply_layout(fig_eq, "", 260)
    fig_eq.update_layout(
        barmode="overlay",
        xaxis_title="Equity Value (млн руб.)",
        yaxis_title="Частота",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Scatter: WACC vs EV ─────────────────────────────────────────────────
    st.markdown("#### 🔍 Чувствительность: ключевые параметры vs EV")
    tabs_inner = st.tabs(["WACC vs EV","Rev Growth vs EV","EBITDA Margin vs EV","Корреляция параметров"])

    sample_n = min(3000, p["n_sim"])
    idx      = np.random.choice(len(ev), sample_n, replace=False)

    with tabs_inner[0]:
        fig_sc = go.Figure(go.Scatter(
            x=mc["waccs"][idx]*100, y=ev[idx],
            mode="markers",
            marker=dict(color=ev[idx], colorscale="Viridis",
                        size=3, opacity=0.5, showscale=True,
                        colorbar=dict(title="EV")),
        ))
        apply_layout(fig_sc, "", 300)
        fig_sc.update_layout(
            xaxis_title="WACC (%)", yaxis_title="EV (млн руб.)"
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with tabs_inner[1]:
        fig_sc2 = go.Figure(go.Scatter(
            x=mc["rev_gs"][idx]*100, y=ev[idx],
            mode="markers",
            marker=dict(color=ev[idx], colorscale="Plasma",
                        size=3, opacity=0.5, showscale=True),
        ))
        apply_layout(fig_sc2, "", 300)
        fig_sc2.update_layout(
            xaxis_title="Revenue Growth (%)", yaxis_title="EV (млн руб.)"
        )
        st.plotly_chart(fig_sc2, use_container_width=True)

    with tabs_inner[2]:
        fig_sc3 = go.Figure(go.Scatter(
            x=mc["ems"][idx]*100, y=ev[idx],
            mode="markers",
            marker=dict(color=ev[idx], colorscale="RdYlGn",
                        size=3, opacity=0.5, showscale=True),
        ))
        apply_layout(fig_sc3, "", 300)
        fig_sc3.update_layout(
            xaxis_title="EBITDA Margin (%)", yaxis_title="EV (млн руб.)"
        )
        st.plotly_chart(fig_sc3, use_container_width=True)

    with tabs_inner[3]:
        # Корреляция параметров с EV
        params_df = pd.DataFrame({
            "WACC":          mc["waccs"],
            "Rev Growth":    mc["rev_gs"],
            "EBITDA Margin": mc["ems"],
            "Terminal GR":   mc["tgrs"],
            "EV":            ev,
        })
        corr_mc = params_df.corr().round(3)
        fig_corr = px.imshow(
            corr_mc, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, text_auto=".2f",
            template=TPL, height=320,
        )
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── VaR / CVaR таблица ──────────────────────────────────────────────────
    st.markdown("#### ⚠️ Value at Risk & Expected Shortfall (CVaR)")
    var_levels = [1, 5, 10]
    var_rows = []
    for alpha in var_levels:
        var_ev  = ep[alpha]
        cvar_ev = float(ev[ev <= var_ev].mean()) if (ev <= var_ev).sum() > 0 else var_ev
        var_eq  = qp[alpha]
        cvar_eq = float(eq[eq <= var_eq].mean()) if (eq <= var_eq).sum() > 0 else var_eq
        loss_ev = ep[50] - var_ev
        var_rows.append({
            "Уровень": f"VaR {alpha}%",
            "EV (млн)":       fmt(var_ev),
            "EV CVaR (млн)":  fmt(cvar_ev),
            "Потери vs медиана EV": fmt(-loss_ev) if loss_ev > 0 else "—",
            "Equity (млн)":   fmt(var_eq),
            "Equity CVaR":    fmt(cvar_eq),
        })
    st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

    # ── Кумулятивное распределение (CDF) ────────────────────────────────────
    st.markdown("#### 📈 Кумулятивное распределение EV (CDF)")
    ev_sorted = np.sort(ev)
    cdf       = np.arange(1, len(ev_sorted)+1) / len(ev_sorted)
    fig_cdf   = go.Figure()
    fig_cdf.add_trace(go.Scatter(
        x=ev_sorted, y=cdf*100, mode="lines",
        line=dict(color=C["blue"], width=2.5), name="CDF",
    ))
    # Зоны
    for pct_v, col, lbl in [(5, C["red"],"VaR 5%"),(50,C["green"],"Медиана"),(95,C["purple"],"P95")]:
        fig_cdf.add_vline(
            x=ep[pct_v], line_color=col, line_dash="dash", line_width=1.5,
            annotation_text=lbl, annotation_font=dict(size=10, color=col),
        )
    apply_layout(fig_cdf, "CDF Enterprise Value", 300)
    fig_cdf.update_layout(
        xaxis_title="EV (млн руб.)",
        yaxis_title="Вероятность (%)",
        showlegend=False,
    )
    st.plotly_chart(fig_cdf, use_container_width=True)

    # ── Экспорт ─────────────────────────────────────────────────────────────
    st.markdown("---")
    export_df = pd.DataFrame({
        "Перцентиль":     [f"P{p_}" for p_ in [1,5,10,25,50,75,90,95,99]],
        "EV (млн)":       [round(ep[p_],2) for p_ in [1,5,10,25,50,75,90,95,99]],
        "Equity (млн)":   [round(qp[p_],2) for p_ in [1,5,10,25,50,75,90,95,99]],
    })
    xlsx_mc = to_xlsx({
        "MC Summary": export_df,
        "MC Raw (sample)": pd.DataFrame({
            "EV": ev[:5000], "Equity": eq[:5000],
            "WACC": mc["waccs"][:5000],
            "RevGrowth": mc["rev_gs"][:5000],
            "EBITDAMargin": mc["ems"][:5000],
        })
    })
    st.download_button(
        "📥 Скачать Monte Carlo результаты (XLSX)",
        xlsx_mc, "monte_carlo.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

def tab_ml(df: pd.DataFrame, p: Dict) -> None:
    """ML прогноз Revenue с TimeSeriesSplit и feature importance."""
    st.markdown(
        f'<div class="info-box">🤖 {p["mdl_type"]} · TimeSeriesSplit (3 фолда) · '
        f'Прогноз {p["n_fore"]} кварталов · Целевая: Revenue</div>',
        unsafe_allow_html=True,
    )
    if len(df) < 8:
        st.warning("Нужно минимум 8 периодов."); return

    with st.spinner("Обучение..."):
        try:
            mdl, sc, dm, feats, met = train_ml(df, p["mdl_type"])
        except Exception as e:
            st.error(f"ML error: {e}"); return

    feat_labels = ["Тренд t","Квартал","Год","Lag 1","Lag 2","Lag 4","MA4","YoY%"]

    c1,c2,c3 = st.columns(3)
    c1.metric("📏 MAE (CV)",  f"{met['MAE']:,.1f} млн")
    c2.metric("📐 RMSE (CV)", f"{met['RMSE']:,.1f} млн")
    c3.metric("🎯 R² (CV)",   f"{met['R2']:.4f}")

    try:
        fc = forecast_ml(mdl, sc, dm, feats, p["n_fore"])
    except Exception as e:
        st.error(f"Forecast error: {e}"); return

    # График
    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(
        x=df["Date"], y=df["Revenue"],
        name="Факт", line=dict(color=C["blue"], width=2.5),
        mode="lines+markers", marker=dict(size=4),
    ))

    # CI ±1.96*RMSE
    ci = met["RMSE"] * 1.96
    fig_ml.add_trace(go.Scatter(
        x=pd.concat([fc["Date"], fc["Date"].iloc[::-1]]),
        y=pd.concat([fc["Revenue"]+ci, (fc["Revenue"]-ci).iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(245,158,11,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", showlegend=True,
    ))
    fig_ml.add_trace(go.Scatter(
        x=fc["Date"], y=fc["Revenue"],
        name="Прогноз", mode="lines+markers",
        line=dict(color=C["amber"], width=2.5, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
    ))
    fig_ml.add_vline(
        x=df["Date"].max(), line_dash="dot",
        line_color="rgba(255,255,255,0.25)", line_width=1.5,
    )
    apply_layout(fig_ml, f"Revenue Forecast — {p['mdl_type']}", 380)
    fig_ml.update_layout(yaxis_title="млн руб.")
    st.plotly_chart(fig_ml, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### 📋 Прогноз по периодам")
        fc_show = fc.copy()
        fc_show["Период"] = fc_show["Date"].dt.strftime("%Y-Q") + \
                            fc_show["Date"].dt.quarter.astype(str)
        fc_show["Revenue (млн)"] = fc_show["Revenue"].map("{:,.1f}".format)
        fc_show["YoY% (прогноз)"] = ""
        st.dataframe(fc_show[["Период","Revenue (млн)"]],
                     use_container_width=True, hide_index=True)

    with cr:
        if p["mdl_type"] == "Random Forest Regressor":
            st.markdown("#### 🔑 Feature Importance")
            fi = pd.DataFrame({"Признак": feat_labels, "Важность": mdl.feature_importances_})
            fi = fi.sort_values("Важность")
            fig_fi = go.Figure(go.Bar(
                x=fi["Важность"], y=fi["Признак"],
                orientation="h", marker_color=C["blue"],
                text=[f"{v:.3f}" for v in fi["Важность"]],
                textposition="outside",
            ))
            apply_layout(fig_fi, "", 280)
            fig_fi.update_layout(xaxis_title="Важность",
                                 yaxis=dict(color="#8b95aa"),
                                 showlegend=False,
                                 margin=dict(r=60))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.markdown("#### 📊 Коэффициенты LR")
            coef_df = pd.DataFrame({"Признак": feat_labels, "Коэф.": mdl.coef_})
            coef_df = coef_df.sort_values("Коэф.", key=abs)
            fig_cf = go.Figure(go.Bar(
                x=coef_df["Коэф."], y=coef_df["Признак"],
                orientation="h",
                marker_color=[C["green"] if v >= 0 else C["red"]
                              for v in coef_df["Коэф."]],
            ))
            apply_layout(fig_cf, "", 280)
            fig_cf.update_layout(showlegend=False)
            st.plotly_chart(fig_cf, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def tab_scenarios(df: pd.DataFrame, p: Dict) -> None:
    """Сценарный анализ: Pessimistic / Base / Optimistic + Tornado + Sensitivity."""
    with st.expander("⚙️ Настройка отклонений", expanded=False):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown("**📉 Pessimistic**")
            p_rg  = st.slider("ΔRevG (%)", -20, 0, -6, key="sc_prg") / 100
            p_em  = st.slider("ΔEBITDA% (%)", -15, 0, -5, key="sc_pem") / 100
            p_wc  = st.slider("ΔWACC (%)",  0, 8, 3, key="sc_pwc") / 100
        with pc3:
            st.markdown("**📈 Optimistic**")
            o_rg  = st.slider("ΔRevG (%)", 0, 20, 6, key="sc_org") / 100
            o_em  = st.slider("ΔEBITDA% (%)", 0, 15, 5, key="sc_oem") / 100
            o_wc  = st.slider("ΔWACC (%)", -8, 0, -2, key="sc_owc") / 100

    scenarios = {
        "Pessimistic": dict(
            rev_g=p["rev_g"]+p_rg, ebitda_m=p["ebitda_m"]+p_em,
            wacc=p["wacc"]+p_wc
        ),
        "Base": dict(rev_g=p["rev_g"], ebitda_m=p["ebitda_m"], wacc=p["wacc"]),
        "Optimistic": dict(
            rev_g=p["rev_g"]+o_rg, ebitda_m=p["ebitda_m"]+o_em,
            wacc=max(0.02, p["wacc"]+o_wc)
        ),
    }

    rows, ev_vals = [], {}
    for name, sc in scenarios.items():
        try:
            _, s = run_dcf(df, sc["wacc"], p["tgr"], sc["rev_g"],
                           sc["ebitda_m"], p["horizon"])
            ev_vals[name] = s["ev"]
            rows.append({
                "Сценарий":       name,
                "Rev Growth":     f"{sc['rev_g']*100:.1f}%",
                "EBITDA Margin":  f"{sc['ebitda_m']*100:.1f}%",
                "WACC":           f"{sc['wacc']*100:.1f}%",
                "EV (млн)":       fmt(s["ev"]),
                "Equity (млн)":   fmt(s["eq"]),
                "EV/Revenue":     f"{s['ev_rev']:.1f}x",
                "_ev": s["ev"], "_eq": s["eq"],
            })
        except Exception as e:
            st.error(f"{name}: {e}")

    if not rows: return

    res_df = pd.DataFrame(rows)
    st.dataframe(
        res_df.drop(columns=["_ev","_eq"]),
        use_container_width=True, hide_index=True,
    )

    cl, cr = st.columns(2)
    sc_names   = res_df["Сценарий"].tolist()
    sc_colors  = [C["red"], C["blue"], C["green"]]
    ev_list    = res_df["_ev"].tolist()
    eq_list    = res_df["_eq"].tolist()

    with cl:
        fig_ev = go.Figure(go.Bar(
            x=sc_names, y=ev_list,
            marker_color=sc_colors,
            text=[fmt(v) for v in ev_list],
            textposition="outside",
        ))
        apply_layout(fig_ev, "Enterprise Value по сценариям", 300)
        fig_ev.update_layout(showlegend=False, yaxis_title="млн руб.")
        st.plotly_chart(fig_ev, use_container_width=True)

    with cr:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="EV",     x=sc_names, y=ev_list, marker_color=C["blue"]))
        fig_cmp.add_trace(go.Bar(
            name="Equity", x=sc_names, y=eq_list, marker_color=C["green"]))
        apply_layout(fig_cmp, "EV vs Equity", 300)
        fig_cmp.update_layout(barmode="group", yaxis_title="млн руб.")
        st.plotly_chart(fig_cmp, use_container_width=True)

    # Tornado
    st.markdown("#### 🌪️ Tornado Chart")
    base_ev = ev_vals.get("Base", 1)
    t_rows = [
        ("Pessimistic", ev_vals.get("Pessimistic", base_ev) - base_ev),
        ("Base",        0),
        ("Optimistic",  ev_vals.get("Optimistic", base_ev)  - base_ev),
    ]
    t_df = pd.DataFrame(t_rows, columns=["Сценарий","ΔEV"])
    fig_t = go.Figure(go.Bar(
        x=t_df["ΔEV"], y=t_df["Сценарий"],
        orientation="h",
        marker_color=[C["red"] if v < 0 else (C["blue"] if v==0 else C["green"])
                      for v in t_df["ΔEV"]],
        text=[fmt(v) if v != 0 else "Base" for v in t_df["ΔEV"]],
        textposition="outside",
    ))
    fig_t.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    apply_layout(fig_t, "Tornado: ΔEV vs Base Case", 220)
    fig_t.update_layout(showlegend=False, xaxis_title="ΔEV (млн руб.)",
                        margin=dict(r=80))
    st.plotly_chart(fig_t, use_container_width=True)

    # Экспорт
    st.markdown("---")
    xlsx_sc = to_xlsx({"Scenarios": res_df.drop(columns=["_ev","_eq"])})
    st.download_button(
        "📥 Скачать Scenario Analysis (XLSX)",
        xlsx_sc, "scenario_analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Точка входа."""
    st.markdown(f"""
    <div class="fin-header">
      <div style="display:flex;align-items:center;gap:16px;">
        <div style="width:44px;height:44px;background:linear-gradient(135deg,#1d4ed8,#7c3aed);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:22px;">
          📊
        </div>
        <div>
          <div class="fin-title">{APP_TITLE}</div>
          <div class="fin-sub">{APP_SUBTITLE}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    p = sidebar()

    # Данные
    df: Optional[pd.DataFrame] = None
    if p["uploaded"]:
        df, err = load_file(p["uploaded"])
        if err:
            st.error(f"❌ {err}"); df = None
        else:
            st.success(f"✅ Загружен: **{p['uploaded'].name}** — {len(df)} записей")

    if df is None:
        df = generate_synthetic()
        st.info("📌 **Синтетические данные** (5 лет, 20 кварталов). Загрузите CSV/XLSX в сайдбаре.")

    if df.empty:
        st.error("Данные пусты."); return

    # Вкладки
    t1,t2,t3,t4,t5,t6 = st.tabs([
        "📊 Dashboard",
        "🔍 EDA",
        "💰 DCF Valuation",
        "🎲 Monte Carlo",
        "🤖 Machine Learning",
        "🎭 Scenario Analysis",
    ])

    with t1: tab_dashboard(df)
    with t2: tab_eda(df)
    with t3: tab_dcf(df, p)
    with t4: tab_montecarlo(df, p)
    with t5: tab_ml(df, p)
    with t6: tab_scenarios(df, p)


if __name__ == "__main__":
    main()
