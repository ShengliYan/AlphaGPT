import plotly.express as px
import plotly.graph_objects as go


def plot_pnl_distribution(portfolio_df):
    if portfolio_df.empty:
        return go.Figure()

    colors = ["#00FF00" if x > 0 else "#FF0000" for x in portfolio_df["pnl_pct"]]
    fig = go.Figure(data=[go.Bar(
        x=portfolio_df["symbol"],
        y=portfolio_df["pnl_pct"],
        marker_color=colors
    )])
    fig.update_layout(
        title="Current Positions PnL %",
        yaxis_tickformat=".2%",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def plot_market_scatter(market_df):
    if market_df.empty:
        return go.Figure()

    size_col = "quote_volume" if "quote_volume" in market_df.columns else "volume"
    fig = px.scatter(
        market_df,
        x="quote_volume" if "quote_volume" in market_df.columns else "volume",
        y="close" if "close" in market_df.columns else "volume",
        size=size_col,
        color="underlying" if "underlying" in market_df.columns else "symbol",
        hover_name="symbol",
        log_x=True,
        title="TradFi USDT perps (size = quote volume)",
        template="plotly_dark"
    )
    return fig
