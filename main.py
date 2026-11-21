import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

# ── 한글 폰트 CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── CSV 읽기 (인코딩 자동 시도) ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """바이트 데이터 → utf-8-sig → cp949 순서로 읽기"""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def read_csv(file) -> pd.DataFrame:
    """file_uploader 객체 → getvalue()로 한 번만 읽기"""
    if file is None:
        return pd.DataFrame()
    return read_csv_bytes(file.getvalue())


# ── 열 이름 표준화 (KeyError 방지, 핵심!) ──────────────────────────────────────
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df


# ── 공통 스타일 적용 함수 ─────────────────────────────────────────────────────
def apply_style(fig: go.Figure):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
    )


# ── 추세선 추가 (np.polyfit 직접 사용, statsmodels 불필요) ───────────────────
def add_trendline(fig: go.Figure, x, y, color="red"):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    xs = np.array([df["x"].min(), df["x"].max()])
    fig.add_trace(go.Scatter(
        x=xs,
        y=slope * xs + intercept,
        mode="lines",
        line=dict(color=color, dash="dash", width=2),
        name=f"추세선 (r={df['x'].corr(df['y']):.2f})",
    ))


# ── 생중량 분포 플롯 (박스플롯 + 개별 점) ─────────────────────────────────────
def dist_plot(df: pd.DataFrame):
    s = df["생중량"].dropna()
    if len(s) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=s,
        name="생중량",
        fillcolor="#6366f1",
        boxmean=True,
        line=dict(color="#4f46e5"),
    ))
    fig.add_trace(go.Scatter(
        x=np.ones(len(s)) * 1,
        y=s,
        mode="markers",
        marker=dict(size=5, color="#6366f1", opacity=0.6),
        name="개별 값",
    ))
    fig.update_xaxes(title_text="")
    apply_style(fig)
    return fig


# ──────────────────────────── 사이드바 ─────────────────────────────────────────
st.sidebar.title("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")
growth_col_select = st.sidebar.multiselect(
    "📏 생육 측정 항목 (그래프에 표시)",
    ["잎 수", "지상부 길이", "지하부 길이"],
    default=["잎 수", "지상부 길이", "지하부 길이"],
)

# ── 데이터 로딩 (try/except로 안전장치) ───────────────────────────────────────
with st.sidebar.spinner("데이터를 읽는 중…"):
    try:
        env_raw = read_csv(env_file)
        env_df = std_cols(env_raw) if not env_raw.empty else pd.DataFrame()
    except Exception:
        env_df = pd.DataFrame()

    try:
        growth_raw = read_csv(growth_file)
        growth_df = std_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()
    except Exception:
        growth_df = pd.DataFrame()

# 학교명 추출 (파일명에서)
school_name = "우리 학교"
if env_file is not None:
    school_name = os.path.splitext(os.path.basename(env_file.name))[0]
    school_name = school_name.replace("_환경데이터", "").replace("_환경", "")
elif growth_file is not None:
    school_name = os.path.splitext(os.path.basename(growth_file.name))[0]
    school_name = school_name.replace("_생육결과데이터", "").replace("_생육", "")

st.sidebar.success(f"🏫 {school_name}")

# ──────────────────────────── 탭 ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — 실험 개요
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    try:
        st.header(f"🌱 {school_name} 극지식물 개체별 생육 차이 분석")

        # ── 연구 배경 및 목적 ──────────────────────────────────────────────────
        with st.expander("🔍 연구 배경 및 목적"):
            st.markdown("""
**연구 배경**
극지식물은 낮은 온도, 적은 수분, 영양분이 부족한 환경에서도 살아남는 강한 식물입니다.
우리 학교에서는 이런 극지식물을 직접 키우며 환경 데이터(온도·습도·pH·EC 등)와
개체별 생육 결과(잎 수, 길이, 생중량)를 기록하고 있습니다.

**분석 목적**
- 개체마다 생육 차이가 얼마나 나는지 수치로 확인한다.
- 잎 수·지상부 길이와 생중량이 서로 관련이 있는지 알아본다.
- 생중량이 특히 크거나 작은 개체를 찾아 생육 차이의 원인을 생각한다.

**통계 용어 안내**
- **평균**: 여러 값을 더해 개수로 나눈 값. 대표적인 '중간 크기'를 나타냅니다.
- **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 수치입니다.
  값이 클수록 개체 간 차이가 큽니다.
- **변동계수(CV)**: 표준편차를 평균으로 나눈 값(%)입니다.
  평균이 다른 두 그룹을 비교할 때 유용합니다.
- **상관계수(r)**: 두 값이 함께 움직이는 정도를 -1 ~ 1 사이의 숫자로 나타냅니다.
  1에 가까우면 함께 증가하고, -1에 가까우면 한쪽이 증가할 때 다른 쪽이 감소합니다.
""")

        # ── 데이터 요약 ────────────────────────────────────────────────────────
        st.subheader("📋 데이터 요약")

        total_growth = len(growth_df)
        has_env = not env_df.empty and "time" in env_df.columns

        if has_env:
            env_times = env_df["time"].dropna()
            if len(env_times) > 0:
                period = f"{env_times.min().strftime('%Y-%m-%d')} ~ {env_times.max().strftime('%Y-%m-%d')}"
            else:
                period = "측정 기간 정보 없음"
        else:
            period = "환경 데이터 없음"

        summary_rows = []
        summary_rows.append({"항목": "총 개체 수",
                             "값": f"{total_growth}개체" if total_growth > 0 else "없음"})
        summary_rows.append({"항목": "측정 기간", "값": period})
        env_cols_for_summary = [c for c in ["temperature", "humidity", "ph", "ec"]
                                if c in env_df.columns]
        summary_rows.append({
            "항목": "환경 측정 항목",
            "값": ", ".join(env_cols_for_summary) if env_cols_for_summary else "없음",
        })
        growth_cols_for_summary = [c for c in ["잎 수", "지상부 길이", "지하부 길이"]
                                   if c in growth_df.columns]
        summary_rows.append({
            "항목": "생육 측정 항목",
            "값": ", ".join(growth_cols_for_summary) if growth_cols_for_summary else "없음",
        })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, width="stretch", use_container_width=True)

        # ── 주요 지표 카드 4개 ──────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        if total_growth > 0 and "생중량" in growth_df.columns:
            gw = growth_df["생중량"].dropna()
            avg_gw = gw.mean() if len(gw) > 0 else 0
            col1.metric("총 개체 수", f"{total_growth}개체")
            col4.metric("평균 생중량", f"{avg_gw:.2f} g")
        else:
            col1.metric("총 개체 수", "데이터 없음")
            col4.metric("평균 생중량", "데이터 없음")

        if has_env and "temperature" in env_df.columns:
            tmp = env_df["temperature"].dropna()
            col2.metric("평균 온도",
                        f"{tmp.mean():.1f} °C" if len(tmp) > 0 else "데이터 없음")
        else:
            col2.metric("평균 온도", "데이터 없음")

        if has_env and "humidity" in env_df.columns:
            hum = env_df["humidity"].dropna()
            col3.metric("평균 습도",
                        f"{hum.mean():.1f} %" if len(hum) > 0 else "데이터 없음")
        else:
            col3.metric("평균 습도", "데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — 환경 데이터
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty or "time" not in env_df.columns:
            st.info("🌡️ 환경 데이터 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터")

            # ── 환경 요약 표 ────────────────────────────────────────────────────
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                        if c in env_df.columns]
            if env_cols:
                st.subheader("📊 환경 요약 (평균 · 최소 · 최대)")
                summary_rows = []
                for c in env_cols:
                    s = env_df[c].dropna()
                    if len(s) > 0:
                        summary_rows.append({
                            "측정 항목": c.capitalize(),
                            "평균": f"{s.mean():.2f}",
                            "최소": f"{s.min():.2f}",
                            "최대": f"{s.max():.2f}",
                            "측정 횟수": str(len(s)),
                        })
                st.dataframe(pd.DataFrame(summary_rows), width="stretch")

            # ── 일별 평균 시계열 ────────────────────────────────────────────────
            env_daily = env_df.set_index("time").resample("D").mean().dropna(how="all")

            has_temp = "temperature" in env_daily.columns
            has_hum = "humidity" in env_daily.columns
            has_ph = "ph" in env_daily.columns
            has_ec = "ec" in env_daily.columns
            if not (has_temp or has_hum or has_ph or has_ec):
                st.info("시계열로 표시할 환경 열이 없습니다.")
            else:
                title_list = []
                if has_temp:
                    title_list.append("온도 변화")
                if has_hum:
                    title_list.append("습도 변화")
                if has_ph:
                    title_list.append("pH 변화")
                if has_ec:
                    title_list.append("EC 변화")
                n_rows = len(title_list)

                fig = make_subplots(
                    rows=n_rows,
                    cols=1,
                    subplot_titles=title_list,
                    vertical_spacing=0.12,
                )

                color_map = {
                    "temperature": "#f97316",
                    "humidity": "#3b82f6",
                    "ph": "#8b5cf6",
                    "ec": "#22c55e",
                }
                row_num = 1
                for col_name in ["temperature", "humidity", "ph", "ec"]:
                    if col_name not in env_daily.columns:
                        continue
                    s = env_daily[col_name].dropna()
                    if len(s) == 0:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=env_daily.index,
                            y=env_daily[col_name],
                            mode="lines",
                            line=dict(color=color_map[col_name], width=2),
                            name=col_name.capitalize(),
                            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                        ),
                        row=row_num,
                        col=1,
                    )
                    row_num += 1

                for r in range(1, n_rows + 1):
                    fig.update_xaxes(title_text="날짜",
                                     row=r, col=1, tickformat="%m-%d")
                    fig.update_yaxes(title_text="", row=r, col=1)

                fig.update_layout(height=max(280, 240 * n_rows),
                                  showlegend=True)
                apply_style(fig)
                st.plotly_chart(fig, width="stretch")

            # ── 원본 테이블 + 다운로드 ──────────────────────────────────────────
            with st.expander("📄 환경 데이터 원본 테이블"):
                st.dataframe(env_df, width="stretch", use_container_width=True)
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 환경 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — 생육 결과
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty or "생중량" not in growth_df.columns or len(growth_df) == 0:
            st.info("📊 생육 결과 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            st.header(f"📊 {school_name} 생육 결과 분석")

            gw_col = growth_df["생중량"].dropna()
            leaf_col = growth_df["잎 수"].dropna() if "잎 수" in growth_df.columns else pd.Series(dtype=float)
            shoot_col = growth_df["지상부 길이"].dropna() if "지상부 길이" in growth_df.columns else pd.Series(dtype=float)
            root_col = growth_df["지하부 길이"].dropna() if "지하부 길이" in growth_df.columns else pd.Series(dtype=float)

            # ── 핵심 결과 카드 ──────────────────────────────────────────────────
            st.subheader("🥇 핵심 결과")
            c_avg, c_max, c_min = st.columns(3)

            avg_gw = gw_col.mean() if len(gw_col) > 0 else 0
            if len(gw_col) > 0:
                max_idx = gw_col.idxmax()
                min_idx = gw_col.idxmin()
                max_val = gw_col.max()
                min_val = gw_col.min()
                max_id = growth_df.loc[max_idx, "개체번호"]
                min_id = growth_df.loc[min_idx, "개체번호"]
            else:
                max_id = min_id = "N/A"
                max_val = min_val = 0

            c_avg.metric("평균 생중량", f"{avg_gw:.2f} g")
            c_max.metric("최대 개체", f"{max_id} ({max_val:.2f} g)")
            c_min.metric("최소 개체", f"{min_id} ({min_val:.2f} g)")

            # ── 개체별 생육 비교 (막대 그래프) ────────────────────────────────────
            st.subheader("📈 개체별 생육 비교")

            show_cols = [c for c in growth_col_select
                         if c in growth_df.columns and growth_df[c].dropna().shape[0] > 0]
            if not show_cols:
                st.info("표시할 생육 측정 항목이 없습니다. 사이드바에서 선택해 주세요.")
            else:
                # 생중량 상위/하위 25%
                top_idx_set = set()
                bot_idx_set = set()
                if len(gw_col) >= 2:
                    q75 = gw_col.quantile(0.75)
                    q25 = gw_col.quantile(0.25)
                    top_idx_set = set(gw_col[gw_col >= q75].index)
                    bot_idx_set = set(gw_col[gw_col <= q25].index)

                for metric_col in show_cols:
                    fig = go.Figure()
                    vals = growth_df[metric_col].dropna()
                    if len(vals) == 0:
                        continue
                    ids = growth_df.loc[vals.index, "개체번호"]

                    bar_colors = []
                    for idx in vals.index:
                        if idx in top_idx_set:
                            bar_colors.append("#22c55e")
                        elif idx in bot_idx_set:
                            bar_colors.append("#ef4444")
                        else:
                            bar_colors.append("#a5b4fc")

                    fig.add_trace(go.Bar(
                        x=ids.astype(str).tolist(),
                        y=vals.values,
                        marker_color=bar_colors,
                        name=metric_col,
                        hovertemplate="%{x}<br>%{y:.1f}<extra></extra>",
                    ))
                    fig.update_xaxes(title_text="개체", tickangle=-45)
                    fig.update_yaxes(title_text=metric_col)
                    apply_style(fig)
                    st.plotly_chart(fig, width="stretch")

            # ── 생중량 분포 ──────────────────────────────────────────────────────
            st.subheader("📦 생중량 분포")
            dp = dist_plot(growth_df)
            if dp is not None:
                st.plotly_chart(dp, width="stretch")
            else:
                st.info("생중량 데이터가 부족해 분포를 표시할 수 없습니다.")

            # ── 상관관계 분석 (산점도 + 추세선) ────────────────────────────────
            st.subheader("🔗 상관관계 분석")

            gw_series = growth_df["생중량"].dropna()
            if len(gw_series) < 2:
                st.info("생중량 데이터가 부족해 상관관계를 분석할 수 없습니다.")
            else:
                corr_pairs = [
                    ("잎 수", "잎 수"),
                    ("지상부 길이", "지상부 길이"),
                ]
                for label, col_name in corr_pairs:
                    if col_name not in growth_df.columns:
                        st.info(
                            f"'{label}' 데이터가 없어 "
                            f"{label} vs 생중량 산점도를 표시할 수 없습니다."
                        )
                        continue
                    xs = growth_df[col_name].dropna()
                    valid_idx = xs.index.intersection(gw_series.index)
                    if len(valid_idx) < 2:
                        st.info(
                            f"{label} 데이터가 부족해 "
                            f"산점도를 표시할 수 없습니다."
                        )
                        continue
                    x_vals = growth_df.loc[valid_idx, col_name]
                    y_vals = growth_df.loc[valid_idx, "생중량"]
                    r = x_vals.corr(y_vals)
                    fig = px.scatter(
                        x=x_vals,
                        y=y_vals,
                        labels={col_name: label, "생중량": "생중량 (g)"},
                    )
                    fig.update_traces(
                        marker=dict(size=8, color="#6366f1", opacity=0.7),
                        showlegend=False,
                    )
                    add_trendline(fig, x_vals, y_vals, color="#ef4444")
                    fig.update_layout(
                        title=f"{label} vs 생중량 (상관계수 r = {r:.2f})",
                    )
                    apply_style(fig)
                    st.plotly_chart(fig, width="stretch")

            # ── 원본 테이블 + 다운로드 ──────────────────────────────────────────
            with st.expander("📄 생육 데이터 원본 테이블"):
                st.dataframe(growth_df, width="stretch", use_container_width=True)
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ 생육 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
