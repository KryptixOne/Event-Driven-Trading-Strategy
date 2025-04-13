import matplotlib.pyplot as plt
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def plot_symbol_and_signals_old(df, show_indicators=False, show_vwap_bands=False):
    """
    Plots the main price series plus the Buy/Sell signals and optionally:
        - RSI, MACD lines, or other indicators (in a separate chart if show_indicators=True).
        - VWAP with upper/lower bands (if show_vwap_bands=True).

    Parameters:
        df (DataFrame): DataFrame that already has columns:
                        ['Open','High','Low','Close','Buy','Sell','EarlyBullFlip','EarlyBearFlip',
                         'ExitLong','ExitShort','VWAP','VWAP_Upper','VWAP_Lower', etc.]
        show_indicators (bool): If True, plot the RSI, MACD, etc. in additional charts.
        show_vwap_bands (bool): If True, plot daily VWAP with upper/lower bands in another chart.
    """

    # --- 1) PRICE & SIGNALS ---
    plt.figure()
    plt.plot(df.index, df['Close'], label='Close')

    # Plot buy/sell signals as scatter points
    buy_idx = df.index[df['Buy'] == True]
    sell_idx = df.index[df['Sell'] == True]

    plt.scatter(buy_idx, df.loc[buy_idx, 'Close'], marker='^', label='Buy Signal')
    plt.scatter(sell_idx, df.loc[sell_idx, 'Close'], marker='v', label='Sell Signal')

    # Plot exit signals as well
    exitLong_idx = df.index[df['ExitLong'] == True]
    exitShort_idx = df.index[df['ExitShort'] == True]
    plt.scatter(exitLong_idx, df.loc[exitLong_idx, 'Close'], marker='v', label='Exit Long')
    plt.scatter(exitShort_idx, df.loc[exitShort_idx, 'Close'], marker='^', label='Exit Short')

    # Plot early reversals
    bullFlip_idx = df.index[df['EarlyBullFlip'] == True]
    bearFlip_idx = df.index[df['EarlyBearFlip'] == True]
    plt.scatter(bullFlip_idx, df.loc[bullFlip_idx, 'Close'], marker='^', label='Early Bull Flip')
    plt.scatter(bearFlip_idx, df.loc[bearFlip_idx, 'Close'], marker='v', label='Early Bear Flip')

    # Optionally, overlay floor/ceiling lines
    if 'Floor' in df.columns and 'Ceiling' in df.columns:
        plt.plot(df.index, df['Floor'], label='Floor')
        plt.plot(df.index, df['Ceiling'], label='Ceiling')

    plt.legend()
    plt.title("Price & Signals")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

    # --- 2) OPTIONAL: INDICATORS ---
    if show_indicators:
        # Example: RSI
        if 'RSI_14' in df.columns:
            plt.figure()
            plt.plot(df.index, df['RSI_14'], label='RSI(14)')
            plt.title("RSI(14)")
            plt.xlabel("Time")
            plt.ylabel("RSI Value")
            plt.legend()
            plt.show()

        # Example: MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            plt.figure()
            plt.plot(df.index, df['MACD'], label='MACD')
            plt.plot(df.index, df['MACD_Signal'], label='Signal')
            plt.title("MACD")
            plt.xlabel("Time")
            plt.ylabel("MACD Value")
            plt.legend()
            plt.show()


    # --- 3) OPTIONAL: VWAP BANDS ---
    if show_vwap_bands and 'VWAP' in df.columns:
        plt.figure()
        plt.plot(df.index, df['Close'], label='Close')
        plt.plot(df.index, df['VWAP'], label='VWAP')
        # If the user computed upper/lower bands
        if 'VWAP_Upper' in df.columns and 'VWAP_Lower' in df.columns:
            plt.plot(df.index, df['VWAP_Upper'], label='VWAP Upper')
            plt.plot(df.index, df['VWAP_Lower'], label='VWAP Lower')
        plt.legend()
        plt.title("Daily VWAP + Bands")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()


def plot_symbol_and_signals(df, show_indicators=False, show_vwap_bands=False, market_hours_only=False):
    """
    Plots price, signals, and optionally indicators or VWAP bands.

    Parameters:
        ...
        market_hours_only (bool): If True, hides premarket, after-hours, and weekends.
    """

    # --- FILTER TO MARKET HOURS ONLY ---
    if market_hours_only:
        # Only weekdays (Mon–Fri)
        df = df[df.index.dayofweek < 5]
        # Filter between 9:30 AM and 4:00 PM (Eastern)
        df = df.between_time("09:30", "16:00")

    # --- 1) PRICE & SIGNALS ---
    plt.figure()
    plt.plot(df.index, df['Close'], label='Close')

    # Signals
    for label, idx, marker in [
        ('Buy Signal', df.index[df['Buy']], '^'),
        ('Sell Signal', df.index[df['Sell']], 'v'),
        ('Exit Long', df.index[df['ExitLong']], 'v'),
        ('Exit Short', df.index[df['ExitShort']], '^'),
        ('Early Bull Flip', df.index[df['EarlyBullFlip']], '^'),
        ('Early Bear Flip', df.index[df['EarlyBearFlip']], 'v'),
    ]:
        plt.scatter(idx, df.loc[idx, 'Close'], label=label, marker=marker)

    if 'Floor' in df.columns:
        plt.plot(df.index, df['Floor'], label='Floor')
    if 'Ceiling' in df.columns:
        plt.plot(df.index, df['Ceiling'], label='Ceiling')

    plt.title("Price & Signals")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # --- 2) INDICATORS ---
    if show_indicators:
        if 'RSI_14' in df.columns:
            plt.figure()
            plt.plot(df.index, df['RSI_14'], label='RSI(14)')
            plt.title("RSI(14)")
            plt.legend()
            plt.show()

        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            plt.figure()
            plt.plot(df.index, df['MACD'], label='MACD')
            plt.plot(df.index, df['MACD_Signal'], label='Signal')
            plt.title("MACD")
            plt.legend()
            plt.show()

    # --- 3) VWAP + BANDS ---
    if show_vwap_bands and 'VWAP' in df.columns:
        plt.figure()
        plt.plot(df.index, df['Close'], label='Close')
        plt.plot(df.index, df['VWAP'], label='VWAP')
        if 'VWAP_Upper' in df.columns:
            plt.plot(df.index, df['VWAP_Upper'], label='VWAP Upper')
        if 'VWAP_Lower' in df.columns:
            plt.plot(df.index, df['VWAP_Lower'], label='VWAP Lower')
        plt.title("VWAP + Bands")
        plt.legend()
        plt.show()



def plot_interactive(df):
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines', name='Close'
    ))

    # Buy/Sell signals
    fig.add_trace(go.Scatter(
        x=df.index[df['Buy']], y=df['Close'][df['Buy']],
        mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)
    ))

    fig.add_trace(go.Scatter(
        x=df.index[df['Sell']], y=df['Close'][df['Sell']],
        mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)
    ))

    # VWAP bands if present
    if 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(dash='dot')))
    if 'VWAP_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_Upper'], name='VWAP Upper', line=dict(dash='dot')))
    if 'VWAP_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_Lower'], name='VWAP Lower', line=dict(dash='dot')))

    # Layout
    fig.update_layout(
        title='Interactive Price Chart with Signals',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        height=600,
        yaxis=dict(autorange=True, fixedrange=False)
    )

    fig.show()


def plot_interactive_with_indicators(df, show_indicators=True, show_vwap_bands=True):
    """
    Plots interactive price chart with Buy/Sell signals, optional indicators and VWAP bands.
    Cumulative Delta Volume is shown as color-coded bars (green/red).
    """
    # Define subplot layout
    row_count = 1 + (3 if show_indicators else 0)
    fig = make_subplots(
        rows=row_count, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2][:row_count],
        vertical_spacing=0.02,
        subplot_titles=["Price", "RSI", "MACD", "Chaikin & CumDelta"][:row_count]
    )

    # --- ROW 1: Price & Signals ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'), row=1, col=1)

    if 'Buy' in df:
        fig.add_trace(go.Scatter(
            x=df.index[df['Buy']], y=df['Close'][df['Buy']],
            mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=10)
        ), row=1, col=1)
    if 'Sell' in df:
        fig.add_trace(go.Scatter(
            x=df.index[df['Sell']], y=df['Close'][df['Sell']],
            mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=10)
        ), row=1, col=1)

    # VWAP & Bands
    if show_vwap_bands and 'VWAP' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(dash='dot')), row=1, col=1)
        if 'VWAP_Upper' in df and 'VWAP_Lower' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_Upper'], name='VWAP Upper', line=dict(dash='dot')), row=1,
                          col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_Lower'], name='VWAP Lower', line=dict(dash='dot')), row=1,
                          col=1)

    # --- ROW 2: RSI ---
    if show_indicators and 'RSI_14' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], name='RSI(14)', line=dict(color='blue')), row=2, col=1)

    # --- ROW 3: MACD ---
    if show_indicators and 'MACD' in df and 'MACD_Signal' in df:
        print("MACD:", type(df['MACD']), df['MACD'].shape)
        print("MACD_Signal:", type(df['MACD_Signal']), df['MACD_Signal'].shape)
        print("MACD nulls:", df['MACD'].isna().sum(), df['MACD_Signal'].isna().sum())

        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='purple')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Signal', line=dict(color='orange')), row=3,
                      col=1)

    # --- ROW 4: Chaikin + CumDelta Volume as bar ---
    if show_indicators:
        if 'Chaikin' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['Chaikin'], name='Chaikin', line=dict(color='teal')), row=4,
                          col=1)
        if 'CumDelta' in df:
            colors = ['green' if val >= 0 else 'red' for val in df['CumDelta'].diff().fillna(0)]
            fig.add_trace(go.Bar(
                x=df.index, y=df['CumDelta'].diff().fillna(0),
                name='Delta Volume', marker_color=colors, opacity=0.6
            ), row=4, col=1)

    # --- Layout tweaks ---
    fig.update_layout(
        height=300 + row_count * 200,
        title="Interactive Chart with Signals & Indicators",
        xaxis_rangeslider_visible=False,
        yaxis=dict(autorange=True, fixedrange=False),
        hovermode='x unified'
    )

    fig.show()



def plot_trades_and_signals(df, trades_df, live_signals_df):
    fig = go.Figure()

    # Plot price
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'))

    # Completed trades
    for _, trade in trades_df.iterrows():
        color = 'green' if trade['side'] == 'LONG' else 'red'
        fig.add_trace(go.Scatter(
            x=[trade['entry_time'], trade['exit_time']],
            y=[trade['entry_price'], trade['exit_price']],
            mode='lines+markers',
            marker=dict(symbol='arrow-bar-up' if trade['side'] == 'LONG' else 'arrow-bar-down', size=10),
            line=dict(dash='dot', color=color),
            name=f"{trade['side']} Trade"
        ))

    # Live signals (including currently open positions)
    for _, signal in live_signals_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[signal['timestamp']],
            y=[df.loc[signal['timestamp'], 'Close']] if signal['timestamp'] in df.index else [None],
            mode='markers+text',
            marker=dict(size=12, color='orange'),
            text=[signal['signal']],
            textposition='top center',
            name='Live Signal'
        ))

    fig.update_layout(
        title="Price Chart with Trades and Signals",
        xaxis=dict(
            title="Time",
            tickformat="%b %d\n%H:%M",  # Adaptive: shows date and hour
            tickangle=0,
            showgrid=True,
            ticklabelmode="instant",  # Show labels directly at timestamp
            ticklabeloverflow="allow",  # Avoid clipping
            ticklabelstep=1,
            rangeslider_visible=False,
            type="date"
        )
        ,
        yaxis_title="Price",
        legend=dict(x=0.01, y=0.99),
        height=600
    )

    shade_after_hours(fig, df)

    fig.show(renderer="browser")


def shade_after_hours(fig, df, market_close="19:30:00", next_open="13:30:00"):
    timestamps = df.index.to_list()
    for i in range(len(timestamps) - 1):
        t1 = timestamps[i]
        t2 = timestamps[i + 1]
        # If the time is 19:30 (market close), shade until next day 13:30
        if t1.time().strftime("%H:%M:%S") == market_close:
            next_day_open = t1 + timedelta(days=1)
            next_open_dt = pd.Timestamp(f"{next_day_open.date()} {next_open}").tz_localize(t1.tz)
            fig.add_vrect(
                x0=t1, x1=next_open_dt,
                fillcolor="lightgray", opacity=0.2,
                layer="below", line_width=0
            )


def plot_combined_dashboard(df, trades_df, equity_curve, live_signals_df=None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            "Price Chart with Trades and Signals",
            "Equity (Left) & Cumulative PnL (Right)"
        ),
        specs=[
            [{}],                       # Row 1: Price chart
            [{"secondary_y": True}]     # Row 2: Equity & PnL
        ]
    )

    # ---- Row 1: Price Chart with Trades ----
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'),
        row=1, col=1
    )

    # Completed trades
    for _, trade in trades_df.iterrows():
        color = 'green' if trade['side'] == 'LONG' else 'red'
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_time'], trade['exit_time']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='lines+markers',
                marker=dict(
                    symbol='arrow-bar-up' if trade['side'] == 'LONG' else 'arrow-bar-down',
                    size=10
                ),
                line=dict(dash='dot', color=color),
                name=f"{trade['side']} Trade"
            ),
            row=1, col=1
        )

    # Live signals
    if live_signals_df is not None:
        for _, signal in live_signals_df.iterrows():
            # Only plot if timestamp in df
            if signal['timestamp'] in df.index:
                fig.add_trace(
                    go.Scatter(
                        x=[signal['timestamp']],
                        y=[df.loc[signal['timestamp'], 'Close']],
                        mode='markers+text',
                        marker=dict(size=12, color='orange'),
                        text=[signal['signal']],
                        textposition='top center',
                        name='Live Signal'
                    ),
                    row=1, col=1
                )

    # ---- Row 2: Equity (Left Axis) & PnL (Right Axis) ----
    # Add Equity
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity Curve',
            line=dict(color='blue')
        ),
        row=2, col=1, secondary_y=False
    )

    # Add PnL on right axis
    if not trades_df.empty:
        # re-sort trades so we can compute cumsum
        trades_sorted = trades_df.sort_values(by='exit_time').reset_index(drop=True)
        trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()

        fig.add_trace(
            go.Scatter(
                x=trades_sorted['exit_time'],
                y=trades_sorted['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative PnL',
                line=dict(color='orange', shape='hv'),  # "hv" = horizontal-then-vertical step
                marker=dict(color='red', size=6)
            ),

        row=2, col=1, secondary_y=True
        )

    # ---- Layout ----
    fig.update_layout(
        height=800,
        title="Live Dashboard: Trades, Equity, and PnL",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )

    # Format axes
    # Row 1 X-axis
    fig.update_xaxes(
        tickformat="%b %d %H:%M",
        tickangle=45,
        showgrid=True,
        rangeslider_visible=False,
        row=1, col=1
    )
    # Row 2 X-axis
    fig.update_xaxes(
        tickformat="%b %d %H:%M",
        tickangle=45,
        showgrid=True,
        row=2, col=1
    )

    # Row 2 left Y-axis (Equity)
    fig.update_yaxes(
        title_text="Equity ($)",
        secondary_y=False,
        row=2, col=1
    )
    # Row 2 right Y-axis (PnL)
    fig.update_yaxes(
        title_text="Cumulative PnL ($)",
        secondary_y=True,
        row=2, col=1
    )

    # Row 1 Y-axis
    fig.update_yaxes(title_text="Price", row=1, col=1)

    fig.show()
