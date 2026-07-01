"""Trade logger for recording all transactions to file."""

import os
import logging
import json
from datetime import datetime

from config import LOG_DIR

logger = logging.getLogger(__name__)


class TradeLogger:
    """Logs trade details to a structured log file."""

    def __init__(self, log_dir: str = LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

    def log_trade(self, trade: dict):
        entry = {'timestamp': datetime.now().isoformat(), **trade}
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def log_order(self, order):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'ORDER',
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'order_type': order.order_type.value,
            'limit_price': order.limit_price,
            'status': order.status.value,
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def log_fill(self, order):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'FILL',
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'fill_price': order.fill_price,
            'commission': order.commission,
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def log_portfolio_snapshot(self, portfolio, date=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'SNAPSHOT',
            'date': str(date) if date else None,
            'cash': portfolio.cash,
            'equity': portfolio.get_equity(),
            'total_return': portfolio.total_return,
            'positions': {
                sym: {
                    'shares': pos.shares,
                    'avg_cost': pos.avg_cost,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                }
                for sym, pos in portfolio.positions.items()
            },
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def get_log_path(self) -> str:
        return self.log_file
