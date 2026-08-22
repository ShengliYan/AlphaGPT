import time

import streamlit as st

from data_service import DashboardService
from visualizer import plot_market_scatter

st.set_page_config(
    page_title="TradFi Perps Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stDataFrame { border: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    return DashboardService()


svc = get_service()

st.sidebar.title("TradFi Perps")
st.sidebar.caption("DuckDB + Binance USD-M TradFi USDT contracts. Research only.")
st.sidebar.markdown("---")

status = svc.get_db_status()
with st.sidebar:
    st.subheader("DuckDB")
    st.code(status["path"], language="text")
    st.metric("Symbols", status["n_symbols"])
    st.metric("OHLCV rows", status["n_bars"])
    st.caption(f"Last bar: {status['last_time']}")
    st.markdown("---")
    if st.button("Refresh Data"):
        st.rerun()

col1, col2, col3, col4 = st.columns(4)
market_df = svc.get_market_overview()
strategy_data = svc.load_strategy_info()

with col1:
    st.metric("Universe", status["n_symbols"])
with col2:
    st.metric("Snapshot rows", len(market_df))
with col3:
    last_qv = float(market_df["quote_volume"].sum()) if not market_df.empty else 0.0
    st.metric("Snapshot quote volume", f"{last_qv:,.0f}")
with col4:
    formula = strategy_data.get("formula") if isinstance(strategy_data, dict) else strategy_data
    st.metric("Strategy", "AlphaGPT-TradFi", help=str(formula))

tab1, tab2 = st.tabs(["TradFi Snapshot", "Logs"])

with tab1:
    st.subheader("USD-M TradFi perps (latest bar)")
    if not market_df.empty:
        st.plotly_chart(plot_market_scatter(market_df), use_container_width=True)
        st.dataframe(market_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No market data in DuckDB. Run `python -m data_pipeline.run_pipeline`.")

with tab2:
    st.subheader("System Logs (Tail 20)")
    logs = svc.get_recent_logs(20)
    if logs:
        st.code("".join(logs), language="text")
    else:
        st.caption("No logs found or log file path incorrect.")

time.sleep(1)
if st.checkbox("Auto-Refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()
