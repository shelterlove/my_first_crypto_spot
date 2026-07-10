# Futures V1 VPS 部署流程

本文档描述当前正式版：原生 Python 框架 + Binance USD-M Futures Testnet +
ETH/BNB long-only 策略。

## 策略口径

- 策略名：`eth_bnb_futures_v1`
- 交易币种：`ETH/USDT`、`BNB/USDT`
- 参考币种：`BTC/USDT`，只用于市场状态判断
- 周期：`1d`
- 执行市场：Binance USD-M Futures Testnet
- 交易模式：One-way Mode
- 保证金：isolated
- 默认执行器：dry-run，必须传 `--execute` 才会下单
- 虚拟账本：`runtime/futures_state.json`

虚拟 sleeve 账本为 ETH 和 BNB 分别维护权益、仓位、标记价和目标 gross。
账户级充值、提现、手续费、资金费率差额平均分配到两个 sleeve；单币种交易
盈亏只进入对应币种 sleeve。

## 安装

```bash
git clone <repo-url> futures_v1
cd futures_v1
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
cp .env.example .env
```

编辑 `.env`：

```bash
DATABASE_URL=postgresql+psycopg2://quant:quant_password@127.0.0.1:5432/quant_db
BINANCE_FUTURES_TESTNET_API_KEY=你的_testnet_key
BINANCE_FUTURES_TESTNET_API_SECRET=你的_testnet_secret
BINANCE_FUTURES_TESTNET_BASE_URL=https://demo-fapi.binance.com
MONITOR_HOST=127.0.0.1
MONITOR_PORT=8765
```

不要使用生产 Binance key。执行器会拒绝非 futures testnet base URL。

## 数据同步

```bash
python scripts/sync_binance_klines.py \
  --symbols BTC/USDT,ETH/USDT,BNB/USDT \
  --timeframe 1d \
  --start 2020-01-01
```

## Dry Run

```bash
python scripts/binance_futures_testnet_executor.py \
  --config configs/backtest_v1.json \
  --exchange-leverage 3 \
  --target-gross-cap 3.00
```

检查输出报告目录：

```bash
ls -lt results/binance_futures_testnet | head
```

确认目标仓位、订单方向、订单数量、杠杆和账户余额都符合预期后，再执行实盘
testnet 下单。

## Execute

```bash
python scripts/binance_futures_testnet_executor.py \
  --config configs/backtest_v1.json \
  --exchange-leverage 3 \
  --target-gross-cap 3.00 \
  --execute
```

`--max-order-usdt` 默认是 `0`，表示不限制单笔订单金额；只有手动小单测试时
才需要传正数。

## Daemon

```bash
tmux new -s futures_daemon
source .venv/bin/activate
python scripts/run_daemon.py \
  --run-at-utc 01:10 \
  --run-on-start \
  --execute \
  --exchange-leverage 3 \
  --target-gross-cap 3.00
```

日志：

```bash
tail -f logs/futures_daemon.log
```

停止：

```bash
tmux kill-session -t futures_daemon
```

## Monitor

```bash
python scripts/build_monitor_dashboard_data.py
python scripts/serve_monitor.py --host 127.0.0.1 --port 8765
```

如果绑定公网地址，必须设置 `MONITOR_USERNAME` 和 `MONITOR_PASSWORD`，否则服务
会拒绝启动。

## Backtest Review

```bash
python scripts/render_strategy_review_chart.py
```

默认完整窗口为 `2020-01-01` 到 `2026-05-18`，输出到：

```text
results/strategy_review/official_v1_full_20200101_20260518/
```

## 测试

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

## 状态文件

正常情况下不要删除 `runtime/futures_state.json`。如果必须重置，先备份：

```bash
mv runtime/futures_state.json runtime/futures_state.backup.$(date +%Y%m%d_%H%M%S).json
```
