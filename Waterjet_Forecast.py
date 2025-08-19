# app.py — Waterjet Forecast（中文版）
# ------------------------------------------------
# - DB 端 1 分钟聚合：降低传输量、加速
# - 两个预测方法：Holt(阻尼) + 历史相似段(Top-K)
# - 侧栏表单：必须点“开始预测”才执行；“恢复默认”可一键恢复默认
# - 新增：顶部“刷新最新值”按钮，可实时更新“最新温度/故障率”

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

from datetime import datetime, time, timedelta, timezone
import pytz

# ============== 基础配置 ==============
API_URL      = "http://61.177.143.140:8086/query"
DB_NAME      = "DataDB"
MEASUREMENT  = "working_set"
READ_TIMEOUT = (10, 90)
SH_TZ        = pytz.timezone("Asia/Shanghai")
Z_FIXED      = 1.96                   # 95% 置信区间系数
MAX_PLOT_POINTS = 2500                # 绘图降采样上限（不影响预测，仅影响显示）

# 传感器映射：id -> 可读名称（可自行扩展/替换）
SENSOR_MAP = {
    "wxa01sd01.calculate.sdjsd0011": "水泵温度 1",
    "wxa01sd01.calculate.sdjsd0012": "水泵温度 2",
}
LABEL_TO_ID = {v: k for k, v in SENSOR_MAP.items()}

# 复用连接
SESSION = requests.Session()

# ============== 工具函数 ==============
def local_now():
    """获取上海本地当前时间（去秒）。"""
    return datetime.now(SH_TZ).replace(second=0, microsecond=0)

def tz_to_plot(s: pd.Series) -> pd.Series:
    """把索引转成上海时区且 naive，Matplotlib 画图更稳。"""
    s = s.copy()
    if hasattr(s.index, "tz"):
        s.index = s.index.tz_convert(SH_TZ).tz_localize(None)
    return s

def downsample_for_plot(s: pd.Series, max_pts=MAX_PLOT_POINTS) -> pd.Series:
    """绘图用轻度下采样（仅影响展示的流畅性，不影响运算精度）"""
    if s.empty or len(s) <= max_pts:
        return s
    step = max(1, len(s) // max_pts)
    return s.iloc[::step]

def fault_rate_from_temp(t: float) -> float:
    """
    故障率计算：
      - 若温度 >= 45 → 100%
      - 若温度 <= 25 → 0%
      - 介于其间按线性比例：(t - 25) / (45 - 25)
    返回 0.0~1.0
    """
    if t is None or pd.isna(t):
        return np.nan
    if t >= 45:
        return 1.0
    if t <= 25:
        return 0.0
    return max(0.0, min(1.0, (t - 25.0) / 20.0))

# ============== 默认参数 ==============
def compute_defaults():
    """构造一套“当前时间向前 2 小时”的默认参数"""
    now_sh   = local_now()
    start_sh = now_sh - timedelta(hours=2)
    return {
        "sensor_label": list(SENSOR_MAP.values())[0],
        "sel_date": now_sh.date(),
        "sh": start_sh.hour, "sm": start_sh.minute,   # start hour/minute
        "eh": now_sh.hour,   "em": now_sh.minute,     # end   hour/minute
        "steps": 30, "pat_len": 12, "top_k": 3, "roll_w": 7,
        "cache_buster": 0,   # 用于强制刷新缓存
    }

# ✅ **关键修复**：在任何 UI 控件之前，确保 session_state 完整初始化
if "inited" not in st.session_state:
    st.session_state.update(compute_defaults())
    st.session_state["inited"] = True

def reset_to_defaults():
    st.session_state.update(compute_defaults())
    st.rerun()

def ss_get(key, default):
    """安全读取 session_state（兜底）。"""
    return st.session_state[key] if key in st.session_state else default

# ============== DB 侧 1 分钟聚合抓取 ==============
@st.cache_data(ttl=120)
def fetch_minutely_last(sensor_id: str, start_sh: datetime, end_sh: datetime, buster: int = 0) -> pd.Series:
    """
    InfluxQL（对 1.8 OSS 友好）：
      - 用 RFC3339 UTC 时间文本（稳定）
      - 在 DB 端做 1 分钟聚合：last(val)
      - 客户端再转回上海时区
      - buster 仅用于参与缓存键，实现“强制刷新”
    """
    start_utc = start_sh.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc   = end_sh.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    q = (
        f'SELECT last("val") AS v FROM "{MEASUREMENT}" '
        f"WHERE time >= '{start_utc}' AND time <= '{end_utc}' "
        f"AND \"id\" = '{sensor_id}' "
        f"GROUP BY time(1m) fill(none)"
    )
    # 把 buster 加入请求参数，进一步避免代理/中间层复用缓存（非必须）
    r = SESSION.get(API_URL, params={"db": DB_NAME, "q": q, "epoch": "ms", "_": buster}, timeout=READ_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    series = payload.get("results", [{}])[0].get("series", [])
    if not series:
        return pd.Series(dtype=float)

    s0 = series[0]
    df = pd.DataFrame(s0["values"], columns=s0["columns"])  # columns: time, v
    if df.empty or "time" not in df or "v" not in df:
        return pd.Series(dtype=float)

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(SH_TZ)
    df = df.dropna(subset=["v"])
    s = df.set_index("time")["v"].astype(float).sort_index()

    # 对齐完整分钟索引（缺失分钟保留 NaN）
    full_idx = pd.date_range(start=start_sh, end=end_sh, freq="T", tz=SH_TZ)
    return s.reindex(full_idx)

# ============== 预测方法 ==============
def ci_from_sigma(center: pd.Series, sigma: float, hist_len: int, z: float = Z_FIXED):
    """基于残差标准差的简化置信区间（步长越远略放宽）。"""
    h = np.arange(1, len(center) + 1)
    widen = np.sqrt(1.0 + h / max(5, hist_len))
    return center - z * sigma * widen, center + z * sigma * widen

@st.cache_data(ttl=120, show_spinner=False)
def holt_damped(y: pd.Series, steps: int):
    """Holt 阻尼趋势（statsmodels；失败时回退为持久化）。"""
    y = y.dropna().sort_index()
    if y.empty or steps <= 0:
        return y.iloc[-0:0], y.iloc[-0:0], y.iloc[-0:0]
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=None)
        fit = model.fit(optimized=True)
        pred = fit.forecast(steps)
        pred.index = pd.date_range(start=y.index[-1] + pd.Timedelta(minutes=1), periods=steps, freq="T")
        resid = (y - fit.fittedvalues.reindex(y.index)).dropna()
        sigma = float(resid.std(ddof=1)) if len(resid) > 3 else 0.0
        lo, up = ci_from_sigma(pred, sigma, len(y), Z_FIXED)
        return pred, lo, up
    except Exception:
        last = float(y.iloc[-1])
        idx = pd.date_range(start=y.index[-1] + pd.Timedelta(minutes=1), periods=steps, freq="T")
        base = pd.Series([last] * steps, index=idx)
        return base, base, base

@st.cache_data(ttl=120, show_spinner=False)
def pattern_match_fast(y: pd.Series, steps: int, pat_len: int = 12, k: int = 3):
    """矢量化 Top-K 历史相似段匹配，形状配准后集成。"""
    y = y.dropna().sort_index()
    arr = y.values.astype(float)
    n = len(arr); need = pat_len + steps
    if steps <= 0 or n <= need:
        idx = pd.date_range(start=y.index[-1] + pd.Timedelta(minutes=1), periods=steps, freq="T")
        base = pd.Series([float(arr[-1])] * steps, index=idx)
        return base, base, base

    P = arr[-pat_len:]
    stride = arr.strides[0]
    m = n - need + 1
    windows = np.lib.stride_tricks.as_strided(arr, shape=(m, pat_len), strides=(stride, stride))

    win_mu = windows.mean(axis=1, keepdims=True)
    win_sd = windows.std(axis=1, keepdims=True); win_sd[win_sd == 0] = 1.0
    win_z  = (windows - win_mu) / win_sd
    Pz = (P - P.mean()) / (P.std() or 1.0)

    # 近似“相关系数”的相似度
    scores = (win_z * Pz).sum(axis=1) / pat_len
    k = max(1, min(k, m))
    top_idx = np.argpartition(-scores, kth=k-1)[:k]

    futures = []
    for sidx in top_idx:
        F = arr[sidx + pat_len : sidx + pat_len + steps]
        Fz = (F - F.mean()) / (F.std() or 1.0)
        F_adj = Fz * (P.std() or 1.0) + P.mean()
        F_adj += (P[-1] - F_adj[0])           # 与最后一个观测值连续
        futures.append(F_adj)

    futures = np.vstack(futures)
    pred  = futures.mean(axis=0)
    sigma = futures.std(axis=0, ddof=1)

    idx = pd.date_range(start=y.index[-1] + pd.Timedelta(minutes=1), periods=steps, freq="T")
    center = pd.Series(pred, index=idx)
    lower  = pd.Series(pred - Z_FIXED * sigma, index=idx)
    upper  = pd.Series(pred + Z_FIXED * sigma, index=idx)
    return center, lower, upper

# ============== 页面 & 侧栏表单（中文） ==============
st.set_page_config(page_title="水刀预测看板", layout="wide")
st.title("水刀预测看板")

# 顶部操作区：刷新按钮（只更新缓存键，必要时把结束时间拉到当前）
cTop1, cTop2, cTop3 = st.columns([1, 1, 6])
if cTop1.button("🔄 刷新最新值", use_container_width=True):
    # 更新结束时间为当前，确保能取到最新数据（如果用户之前选的是历史窗口）
    now_sh = local_now()
    st.session_state["sel_date"] = now_sh.date()
    st.session_state["eh"] = now_sh.hour
    st.session_state["em"] = now_sh.minute
    # 增加缓存破坏因子，强制重新抓数
    st.session_state["cache_buster"] = ss_get("cache_buster", 0) + 1
    st.rerun()

st.sidebar.header("参数设置")

# ---- 传感器选择（带 index 兜底）----
options = list(SENSOR_MAP.values())
default_label = ss_get("sensor_label", options[0])
default_idx = options.index(default_label) if default_label in options else 0
sensor_label = st.sidebar.selectbox("传感器", options, index=default_idx, key="sensor_label")
sensor_id = LABEL_TO_ID[sensor_label]

# ---- 日期 + 时间（小时/分钟拆分）----
sel_date = st.sidebar.date_input("日期", value=ss_get("sel_date", local_now().date()), key="sel_date")
hrs  = list(range(24))
mins = list(range(60))
c1, c2 = st.sidebar.columns(2)
sh = c1.selectbox("开始-小时",   hrs, index=ss_get("sh", (local_now()-timedelta(hours=2)).hour),   key="sh")
sm = c2.selectbox("开始-分钟",   mins, index=ss_get("sm", (local_now()-timedelta(hours=2)).minute), key="sm")
c3, c4 = st.sidebar.columns(2)
eh = c3.selectbox("结束-小时",     hrs, index=ss_get("eh", local_now().hour),                        key="eh")
em = c4.selectbox("结束-分钟",     mins, index=ss_get("em", local_now().minute),                     key="em")

# ---- 预测参数 ----
steps   = st.sidebar.number_input("未来预测分钟数",       min_value=5,  max_value=240,
                                   value=ss_get("steps", 30),   step=5, key="steps")
pat_len = st.sidebar.number_input("相似片段长度(分钟)",   min_value=6,  max_value=120,
                                   value=ss_get("pat_len", 12), step=1, key="pat_len")
top_k   = st.sidebar.number_input("Top-K 相似片段数",     min_value=1,  max_value=10,
                                   value=ss_get("top_k", 3),   step=1, key="top_k")
roll_w  = st.sidebar.number_input("滚动均值窗口(分钟)",   min_value=3,  max_value=30,
                                   value=ss_get("roll_w", 7),  step=1, key="roll_w")

# "恢复默认"按钮 + 占位
cbtn1, cbtn2 = st.sidebar.columns(2)
if cbtn1.button("恢复默认", use_container_width=True):
    reset_to_defaults()
cbtn2.write("")  # 占位，使布局对齐

# ---- 表单提交按钮：必须点“开始预测”才会执行 ----
with st.sidebar.form("params_form", clear_on_submit=False):
    submitted = st.form_submit_button("开始预测", type="primary", use_container_width=True)

# 组装时间窗口（上海时区 tz-aware）
start_sh = SH_TZ.localize(datetime.combine(st.session_state["sel_date"], time(st.session_state["sh"], st.session_state["sm"])))
end_sh   = SH_TZ.localize(datetime.combine(st.session_state["sel_date"], time(st.session_state["eh"], st.session_state["em"])))
if end_sh <= start_sh:
    end_sh = start_sh + timedelta(minutes=1)

# ============== 主流程（仅在 submitted=True 时运行） ==============
if not submitted:
    st.info("请在左侧设置参数后点击 **开始预测**。如需仅更新最新值，请点击顶部的 **刷新最新值**。")
    st.stop()

with st.spinner("正在抓取数据并计算预测..."):
    try:
        ts = fetch_minutely_last(sensor_id, start_sh, end_sh, ss_get("cache_buster", 0))
    except Exception as e:
        st.error(f"数据获取失败：{e}")
        st.stop()

if ts.dropna().empty:
    st.warning("所选时间范围内无数据，请更换时间范围。")
    st.stop()

smoothed = ts.rolling(window=int(roll_w), min_periods=1).mean()

# 预测
holt_c, holt_lo, holt_up = holt_damped(ts, int(steps))
pat_c,  pat_lo,  pat_up  = pattern_match_fast(ts, int(steps), pat_len=int(pat_len), k=int(top_k))

# 转绘图索引并降采样
ts_p, sm_p = tz_to_plot(ts), tz_to_plot(smoothed)
holt_c_p, holt_lo_p, holt_up_p = tz_to_plot(holt_c), tz_to_plot(holt_lo), tz_to_plot(holt_up)
pat_c_p,  pat_lo_p,  pat_up_p  = tz_to_plot(pat_c),  tz_to_plot(pat_lo),  tz_to_plot(pat_up)

ts_p   = downsample_for_plot(ts_p)
sm_p   = downsample_for_plot(sm_p)
holt_c_p, holt_lo_p, holt_up_p = map(downsample_for_plot, (holt_c_p, holt_lo_p, holt_up_p))
pat_c_p,  pat_lo_p,  pat_up_p  = map(downsample_for_plot, (pat_c_p,  pat_lo_p,  pat_up_p))

# 画图（中文）
title = f"{sensor_label} — {start_sh.strftime('%Y-%m-%d %H:%M')} ~ {end_sh.strftime('%Y-%m-%d %H:%M')}"
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(ts_p.index, ts_p.values, label="实际值", lw=1.1, alpha=0.6)
ax.plot(sm_p.index, sm_p.values, label="平滑", lw=1.6)

ax.plot(holt_c_p.index, holt_c_p.values, "--", label="Holt(阻尼)")
ax.fill_between(holt_c_p.index, holt_lo_p.values, holt_up_p.values, alpha=0.12, label="Holt 置信区间")

ax.plot(pat_c_p.index, pat_c_p.values, "--", label=f"历史相似段 (Top-{int(top_k)})")
ax.fill_between(pat_c_p.index, pat_lo_p.values, pat_up_p.values, alpha=0.12, label="相似段 置信区间")

ax.set_title(title); ax.set_xlabel("时间"); ax.set_ylabel("温度 (°C)")
ax.grid(True, alpha=0.3); ax.legend(loc="best")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
st.pyplot(fig, clear_figure=True)

# 快速指标：最新温度 + 故障率
latest_series = tz_to_plot(ts).dropna()
latest_idx = latest_series.index.max() if not latest_series.empty else None
if latest_idx is not None:
    latest_temp = float(ts.dropna().iloc[-1])
    rate = fault_rate_from_temp(latest_temp)
    cA, cB, cC = st.columns([1, 2, 5])
    cA.metric("最新温度 (°C)", f"{latest_temp:.2f}")
    cB.markdown(f"**时间：** {latest_idx.strftime('%Y-%m-%d %H:%M')}")
    if not np.isnan(rate):
        cC.metric("故障率", f"{rate*100:.0f}%")
else:
    st.info("当前窗口暂无可用最新值。")
