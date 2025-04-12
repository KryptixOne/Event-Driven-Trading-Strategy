import pandas as pd

class StrategyInference:
    def __init__(self, df, best_params, signal_func, strategy_func, unit_size=100, initial_cash=100000, position_mode='both'):
        self.df = df.copy()
        self.best_params = best_params
        self.signal_func = signal_func
        self.strategy_func = strategy_func
        self.unit_size = unit_size
        self.initial_cash = initial_cash
        self.position_mode = position_mode

    def run(self):
        signal_keys = {'sma_short_len', 'sma_long_len', 'rsi_len', 'macd_fast', 'macd_slow', 'macd_sig',
                       'chaikin_fast', 'chaikin_slow'}
        strategy_keys = {'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct'}

        signal_params = {k: v for k, v in self.best_params.items() if k in signal_keys}
        strategy_params = {k: v for k, v in self.best_params.items() if k in strategy_keys}

        buy_signal, sell_signal = self.signal_func(self.df.copy(), **signal_params)
        trades =None
        if self.strategy_func.__name__ == 'run_strategy_trailing_stop':
            equity_curve = self.strategy_func(
                buy_signal, sell_signal,
                trailing_stop_pct=strategy_params.get('trailing_stop_pct'),
                position_mode=self.position_mode
            )
        elif self.strategy_func.__name__ == 'run_strategy_hard_stop_profit_taking':
            equity_curve = self.strategy_func(
                buy_signal, sell_signal,
                stop_loss_pct=strategy_params.get('stop_loss_pct'),
                take_profit_pct=strategy_params.get('take_profit_pct')
            )

        elif self.strategy_func.__name__ == 'run_strategy_trailing_stop_with_trades':
            equity_curve, trades = self.strategy_func(
                buy_signal, sell_signal,
                trailing_stop_pct=strategy_params.get('trailing_stop_pct'),
                position_mode=self.position_mode
            )
        else:
            raise ValueError("Unsupported strategy function.")
        if trades is not None:
            return equity_curve, trades
        return equity_curve
