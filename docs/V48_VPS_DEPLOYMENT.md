# V4.8 VPS Deployment

This is the active deployment path. Freqtrade is legacy-only for this strategy.

## Strategy

- Strategy: `v4_8_eth_bnb`
- Trading symbols: `ETH/USDT`, `BNB/USDT`
- Reference symbol: `BTC/USDT`
- Timeframe: `1d`
- Warmup: at least 220 daily candles
- Initial alignment: max 35% per sleeve

## Required On The VPS

- Python 3.10+
- PostgreSQL access to 1d candles for `BTC/USDT`, `ETH/USDT`, `BNB/USDT`
- Binance Spot Testnet API key and secret
- Testnet balances for `USDT`, optionally `ETH` and `BNB`

Do not put production Binance keys in `.env` while using the testnet executor.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `.env` on the VPS. Keep `.env` private; it is ignored by git.

## Smoke Checks

```bash
python -m py_compile \
  scripts/binance_testnet_v48_executor.py \
  scripts/binance_futures_testnet_v48_executor.py \
  scripts/generate_daily_signal.py
python tests/test_v1_smoke.py
```

Generate a local daily signal without placing orders:

```bash
python scripts/generate_daily_signal.py \
  --strategy v4_8_eth_bnb \
  --config configs/backtest_v1.json \
  --output-dir results/daily_signals
```

## Testnet Dry Run

```bash
python scripts/binance_testnet_v48_executor.py \
  --config configs/backtest_v1.json \
  --max-order-usdt 25
```

Dry run writes a JSON report under `results/binance_testnet_v48/`.

Important report fields:

- `requested_notional`: what the strategy/deployment logic requested.
- `notional`: what will actually be sent after safety caps.
- `clip_reason`: why an order was reduced, e.g. `max_order_usdt` or `available_quote`.

## Testnet Execute

```bash
python scripts/binance_testnet_v48_executor.py \
  --config configs/backtest_v1.json \
  --execute \
  --max-order-usdt 25
```

Remove `--max-order-usdt` only after several dry-run and small testnet execute
cycles match expectations.

## Futures Testnet

Use this only when you want to test fractional gross exposure with USD-M
Futures. It is separate from the spot executor.

Key distinction:

- `--exchange-leverage` is the integer leverage setting sent to Binance, for
  example `3`.
- `--target-gross-cap` is the strategy's fractional exposure cap, for example
  `3.00`.
- `--hard-target-gross-limit` is the local hard safety limit. It defaults to
  `3.00` and prevents accidental higher gross exposure even if exchange
  leverage is larger.

That means the exchange account is configured at 3x and the strategy can target
up to 3x notional exposure. Use `--max-order-usdt` during testnet rollout to
avoid jumping to the full target in one order.

Dry run:

```bash
python scripts/binance_futures_testnet_v48_executor.py \
  --config configs/backtest_v1.json \
  --exchange-leverage 3 \
  --target-gross-cap 3.00 \
  --max-order-usdt 25
```

Execute on USD-M Futures Testnet:

```bash
python scripts/binance_futures_testnet_v48_executor.py \
  --config configs/backtest_v1.json \
  --exchange-leverage 3 \
  --target-gross-cap 3.00 \
  --execute \
  --max-order-usdt 25
```

Requirements before execution:

- USD-M Futures Testnet keys in `BINANCE_FUTURES_TESTNET_API_KEY` and
  `BINANCE_FUTURES_TESTNET_API_SECRET`.
- One-way position mode. The executor exits if Hedge Mode is enabled.
- Long-only positions. The executor exits if it detects a short position.
- No unrelated open futures positions by default. Use
  `--allow-other-positions` only when you intentionally share the account.
- Isolated margin. The executor sets isolated margin before order submission in
  execute mode.

## Cron Example

Run once daily after your candle database has the latest completed daily bar:

Spot dry-run:

```cron
15 1 * * * cd /opt/crypto_spot_v1 && . .venv/bin/activate && python scripts/binance_testnet_v48_executor.py --config configs/backtest_v1.json >> logs/v48_testnet.log 2>&1
```

Futures dry-run:

```cron
20 1 * * * cd /opt/crypto_spot_v1 && . .venv/bin/activate && python scripts/binance_futures_testnet_v48_executor.py --config configs/backtest_v1.json --exchange-leverage 3 --target-gross-cap 3.00 --max-order-usdt 25 >> logs/v48_futures_testnet.log 2>&1
```

Use `--execute` only after dry-run reports have been reviewed.

## Web Monitor

Build dashboard data from local result files:

```bash
python scripts/build_monitor_dashboard_data.py
```

Serve the static dashboard on the VPS:

```bash
python -m http.server 8765 --directory web/monitor
```

Open:

```text
http://your-vps-host:8765/
```

If you expose this outside localhost, put it behind SSH tunneling, a firewall,
or a reverse proxy with authentication. The page is read-only, but it displays
positions, order plans, and account-level risk fields.

Optional cron step after the executor:

```cron
25 1 * * * cd /opt/crypto_spot_v1 && . .venv/bin/activate && python scripts/build_monitor_dashboard_data.py >> logs/v48_monitor.log 2>&1
```

## Known Limits

- The current executor submits daily MARKET orders only.
- Intraday shock-ladder behavior from research is not a true intraday live
  engine yet; daily deployment can only approximate the transformed target.
- Spot execution is cash-constrained. Historical research may include gross
  exposure above 1.0, so live spot orders can be clipped.
- Futures execution can express up to 3x gross exposure by default. Liquidation,
  funding, mark-price movement, and exchange risk limits must be reviewed from
  the generated reports before removing order caps.
