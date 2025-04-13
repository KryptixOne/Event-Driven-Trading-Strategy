import numpy as np
import pandas as pd
import itertools
import optuna


class StrategyOptimizer:
    def __init__(self, df,n_trials=1000, unit_size=100, initial_cash=100000):
        self.df = df.copy()
        self.unit_size = unit_size
        self.initial_cash = initial_cash
        self.signal_func = None
        self.cost_func = None
        self.n_trials =n_trials

    def set_signal_function(self, func):
        self.signal_func = func

    def set_cost_function(self, func):
        self.cost_func = func

    def set_strategy_function(self, func):
        self.strategy_func = func

    def run_strategy_hard_stop_profit_taking(self, buy_signal, sell_signal, stop_loss_pct=None, take_profit_pct=None):
        position = 0
        cash = self.initial_cash
        equity_curve = []
        entry_price = None

        for i in range(len(self.df)):
            price = self.df['Close'].iloc[i]

            # --- SL/TP Exit ---
            if position != 0 and entry_price is not None:
                change_pct = (price - entry_price) / entry_price if position > 0 else (
                                                                                                  entry_price - price) / entry_price
                if (stop_loss_pct and change_pct <= -stop_loss_pct) or (
                        take_profit_pct and change_pct >= take_profit_pct):
                    cash += position * price
                    position = 0
                    entry_price = None

            # --- Signal-Based Entry/Exit ---
            elif buy_signal.iloc[i]:
                if position == 0:
                    position = self.unit_size
                    cash -= self.unit_size * price
                    entry_price = price
                elif position < 0:
                    cash += abs(position) * price  # cover short
                    position = self.unit_size
                    cash -= self.unit_size * price
                    entry_price = price

            elif sell_signal.iloc[i]:
                if position == 0:
                    position = -self.unit_size
                    cash += self.unit_size * price
                    entry_price = price
                elif position > 0:
                    cash += position * price  # exit long
                    position = -self.unit_size
                    cash += self.unit_size * price
                    entry_price = price

            equity = cash + position * price
            equity_curve.append(equity)

        return pd.Series(equity_curve, index=self.df.index)

    def run_strategy_trailing_stop(self, buy_signal, sell_signal, trailing_stop_pct=None, position_mode: str = 'both'):
        position = 0
        cash = self.initial_cash
        equity_curve = []
        entry_price = None
        peak_price = None  # track high-water mark for trailing stop
        print(position_mode)

        n = min(len(self.df), len(buy_signal), len(sell_signal))
        df = self.df.iloc[-n:]
        buy_signal = buy_signal.iloc[-n:]
        sell_signal = sell_signal.iloc[-n:]

        for i in range(n):
            price = df['Close'].iloc[i]

            # Trailing stop logic
            if position != 0 and entry_price is not None and trailing_stop_pct is not None:
                if position > 0:
                    peak_price = max(peak_price, price)
                    if price <= peak_price * (1 - trailing_stop_pct):
                        cash += position * price
                        position = 0
                        entry_price = None
                        peak_price = None
                elif position < 0:
                    peak_price = min(peak_price, price)
                    if price >= peak_price * (1 + trailing_stop_pct):
                        cash += position * price
                        position = 0
                        entry_price = None
                        peak_price = None

            # BUY side
            elif buy_signal.iloc[i] and position_mode in {'both', 'long_only'}:
                if position == 0:
                    position = self.unit_size
                    cash -= self.unit_size * price
                    entry_price = price
                    peak_price = price
                elif position < 0:
                    cash += abs(position) * price
                    position = self.unit_size
                    cash -= self.unit_size * price
                    entry_price = price
                    peak_price = price

            # SELL side
            elif sell_signal.iloc[i] and position_mode in {'both', 'short_only'}:
                if position == 0:
                    position = -self.unit_size
                    cash += self.unit_size * price
                    entry_price = price
                    peak_price = price
                elif position > 0:
                    cash += position * price
                    position = -self.unit_size
                    cash += self.unit_size * price
                    entry_price = price
                    peak_price = price

            equity = cash + position * price
            equity_curve.append(equity)

        return pd.Series(equity_curve, index=buy_signal.index)

    def run_strategy_trailing_stop_with_trades(
            self, buy_signal, sell_signal, trailing_stop_pct=None, position_mode='both'
    ):
        position = 0
        cash = self.initial_cash
        equity_curve = []
        entry_price = None
        peak_price = None  # track high-water mark for trailing stop
        trade_log = []  # list of dicts to store each trade event

        n = min(len(self.df), len(buy_signal), len(sell_signal))
        df = self.df.iloc[-n:]
        buy_signal = buy_signal.iloc[-n:]
        sell_signal = sell_signal.iloc[-n:]

        def record_trade(side, entry_price, exit_price, entry_ts, exit_ts):
            """ Helper to append a trade to trade_log """
            if side == 'LONG':
                pnl = exit_price - entry_price
            else:  # side == 'SHORT'
                pnl = entry_price - exit_price
            trade_log.append({
                'side': side,
                'entry_time': entry_ts,
                'entry_price': entry_price,
                'exit_time': exit_ts,
                'exit_price': exit_price,
                'pnl': pnl
            })

        entry_ts = None

        for i in range(n):
            ts = df.index[i]
            price = df['Close'].iloc[i]

            # TRAILING STOP LOGIC
            if position != 0 and entry_price is not None and peak_price is not None and trailing_stop_pct is not None:
                if position > 0:
                    # Update peak if continuing long
                    peak_price = max(peak_price, price)
                    # If price drops below trailing threshold => exit
                    if price <= peak_price * (1 - trailing_stop_pct):
                        record_trade('LONG', entry_price, price, entry_ts, ts)
                        cash += position * price
                        position = 0
                        entry_price = None
                        peak_price = None

                elif position < 0:
                    # Update trough if continuing short
                    peak_price = min(peak_price, price)
                    # If price rises above trailing threshold => exit
                    if price >= peak_price * (1 + trailing_stop_pct):
                        record_trade('SHORT', entry_price, price, entry_ts, ts)
                        cash += position * price
                        position = 0
                        entry_price = None
                        peak_price = None

            # SIGNAL-BASED ENTRY/FLIP
            # BUY side
            elif buy_signal.iloc[i] and position_mode in {'both', 'long_only'}:
                if position == 0:
                    # Enter new long
                    position = self.unit_size
                    cash -= self.unit_size * price
                    entry_price = price
                    peak_price = price
                    entry_ts = ts
                elif position < 0:
                    # Flip from short to long
                    # First record short exit
                    record_trade('SHORT', entry_price, price, entry_ts, ts)
                    cash += abs(position) * price  # cover short
                    # Now open long
                    position = self.unit_size
                    cash -= self.unit_size * price
                    entry_price = price
                    peak_price = price
                    entry_ts = ts

            # SELL side
            elif sell_signal.iloc[i] and position_mode in {'both', 'short_only'}:
                if position == 0:
                    # Enter new short
                    position = -self.unit_size
                    cash += self.unit_size * price
                    entry_price = price
                    peak_price = price
                    entry_ts = ts
                elif position > 0:
                    # Flip from long to short
                    record_trade('LONG', entry_price, price, entry_ts, ts)
                    cash += position * price  # exit long
                    position = -self.unit_size
                    cash += self.unit_size * price
                    entry_price = price
                    peak_price = price
                    entry_ts = ts

            # Update equity
            equity = cash + position * price
            equity_curve.append(equity)

        # FINAL EXIT if still in a position
        if position != 0 and entry_price is not None and len(df) > 0:
            final_price = df['Close'].iloc[-1]
            final_time = df.index[-1]
            if position > 0:
                # close long
                record_trade('LONG', entry_price, final_price, entry_ts, final_time)
                cash += position * final_price
            else:
                # close short
                record_trade('SHORT', entry_price, final_price, entry_ts, final_time)
                cash += position * final_price

            position = 0
            entry_price = None
            peak_price = None

            # last equity
            equity_curve[-1] = cash  # override final equity after forced exit

        equity_series = pd.Series(equity_curve, index=df.index)
        trades_df = pd.DataFrame(trade_log)
        return equity_series, trades_df

    import pandas as pd

    def run_strategy_with_live_signal_output(self, buy_signal, sell_signal, trailing_stop_pct=0.08, unit_size=100,
                                             initial_cash=100000, position_mode='both'):
        position = 0
        cash = initial_cash
        equity_curve = []
        entry_price = None
        peak_price = None
        trade_log = []
        live_signals = []

        n = min(len(self.df), len(buy_signal), len(sell_signal))
        df = self.df.iloc[-n:]
        buy_signal = buy_signal.iloc[-n:]
        sell_signal = sell_signal.iloc[-n:]

        entry_ts = None

        for i in range(n):
            ts = df.index[i]
            price = df['Close'].iloc[i]
            signal_note = None

            # Trailing stop logic
            if position != 0 and entry_price is not None and peak_price is not None:
                if position > 0:
                    peak_price = max(peak_price, price)
                    if price <= peak_price * (1 - trailing_stop_pct):
                        trade_log.append({
                            'side': 'LONG',
                            'entry_time': entry_ts,
                            'entry_price': entry_price,
                            'exit_time': ts,
                            'exit_price': price,
                            'pnl': (price - entry_price)* abs(position)
                        })
                        cash += position * price
                        signal_note = "Exit Long"
                        position = 0
                        entry_price = None
                        peak_price = None

                elif position < 0:
                    peak_price = min(peak_price, price)
                    if price >= peak_price * (1 + trailing_stop_pct):
                        trade_log.append({
                            'side': 'SHORT',
                            'entry_time': entry_ts,
                            'entry_price': entry_price,
                            'exit_time': ts,
                            'exit_price': price,
                            'pnl': (entry_price - price)* abs(position)
                        })
                        cash += position * price
                        signal_note = "Exit Short"
                        position = 0
                        entry_price = None
                        peak_price = None

            # Entry or flip logic
            if buy_signal.iloc[i] and position_mode in {'both', 'long_only'}:
                if position == 0:
                    position = unit_size
                    cash -= unit_size * price
                    entry_price = price
                    peak_price = price
                    entry_ts = ts
                    signal_note = "Enter Long"
                elif position < 0:
                    trade_log.append({
                        'side': 'SHORT',
                        'entry_time': entry_ts,
                        'entry_price': entry_price,
                        'exit_time': ts,
                        'exit_price': price,
                        'pnl': (entry_price - price)* abs(position)
                    })
                    cash -= abs(position) * price
                    position = unit_size
                    cash -= unit_size * price
                    entry_price = price
                    peak_price = price
                    entry_ts = ts
                    signal_note = "Flip to Long"

            elif sell_signal.iloc[i] and position_mode in {'both', 'short_only'}:
                if position == 0:
                    position = -unit_size
                    cash += unit_size * price
                    entry_price = price
                    peak_price = price
                    entry_ts = ts
                    signal_note = "Enter Short"
                elif position > 0:
                    trade_log.append({
                        'side': 'LONG',
                        'entry_time': entry_ts,
                        'entry_price': entry_price,
                        'exit_time': ts,
                        'exit_price': price,
                        'pnl': (price - entry_price)* abs(position)
                    })
                    cash += position * price
                    position = -unit_size
                    cash += unit_size * price
                    entry_price = price
                    peak_price = price
                    entry_ts = ts
                    signal_note = "Flip to Short"

            # Update equity
            equity = cash + position * price
            equity_curve.append(equity)
            live_signals.append((ts, signal_note))

        equity_series = pd.Series(equity_curve, index=df.index)
        trades_df = pd.DataFrame(trade_log)
        live_signals_df = pd.DataFrame(live_signals, columns=['timestamp', 'signal']).dropna()

        return equity_series, trades_df, live_signals_df

    def optimize(self, param_grid):
        signal_keys = {'sma_short_len', 'sma_long_len', 'rsi_len', 'macd_fast', 'macd_slow', 'macd_sig',
                       'chaikin_fast', 'chaikin_slow'}
        strategy_keys = {'stop_loss_pct', 'take_profit_pct'}
        assert self.signal_func is not None, "Signal function not set."
        assert self.cost_func is not None, "Cost function not set."

        keys, values = zip(*param_grid.items())
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f'length of all combinations is {len(all_combinations)}')
        best_score = -np.inf
        best_params = None
        best_curve = None

        for params in all_combinations:

            signal_params = {k: v for k, v in params.items() if k in signal_keys}
            strategy_params = {k: v for k, v in params.items() if k in strategy_keys}

            buy, sell = self.signal_func(self.df.copy(), **signal_params)
            equity_curve = self.run_strategy(buy, sell, **strategy_params)

            #equity_curve = self.run_strategy(buy, sell)
            score = self.cost_func(equity_curve)
            print(score)

            if score > best_score:
                best_score = score
                best_params = params
                best_curve = equity_curve

        return best_params, best_score, best_curve

    def optimize_with_optuna(self, param_bounds, direction="maximize"):
        assert self.signal_func is not None, "Signal function not set."
        assert self.cost_func is not None, "Cost function not set."

        best_score = -np.inf
        best_params = None
        best_curve = None

        def objective(trial):
            nonlocal best_score, best_params, best_curve
            signal_keys = {'sma_short_len', 'sma_long_len', 'rsi_len', 'macd_fast', 'macd_slow', 'macd_sig',
                           'chaikin_fast', 'chaikin_slow'}
            if self.strategy_func == self.run_strategy_hard_stop_profit_taking:
                strategy_keys = {'stop_loss_pct', 'take_profit_pct'}
            elif self.strategy_func == self.run_strategy_trailing_stop:
                strategy_keys = {'trailing_stop_pct'}
            else:
                strategy_keys = set()

            params = {}
            for key, bounds in param_bounds.items():
                kind = bounds[0]
                if kind == "int":
                    params[key] = trial.suggest_int(key, bounds[1], bounds[2], step=bounds[3])
                elif kind == "float":
                    if len(bounds) == 4:
                        params[key] = trial.suggest_float(key, bounds[1], bounds[2], step=bounds[3])
                    else:
                        params[key] = trial.suggest_float(key, bounds[1], bounds[2])

            if 'sma_short_len' in params and 'sma_long_len' in params:
                if params['sma_short_len'] >= params['sma_long_len']:
                    raise optuna.exceptions.TrialPruned()



            signal_params = {k: v for k, v in params.items() if k in signal_keys}
            strategy_params = {k: v for k, v in params.items() if k in strategy_keys}

            buy, sell = self.signal_func(self.df.copy(), **signal_params)
            equity = self.run_strategy_trailing_stop(buy, sell, **strategy_params)

            score = self.cost_func(equity)

            if score > best_score:
                best_score = score
                best_params = params
                best_curve = equity

            return score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=self.n_trials)

        return best_params, best_score, best_curve, study
