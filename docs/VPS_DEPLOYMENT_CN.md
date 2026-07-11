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

状态文件会绑定 testnet API key 指纹、账户 alias 和配置 hash。状态缺失且已有
非零持仓时执行器会拒绝继续；完成账本审计后才能一次性使用
`--allow-nonzero-bootstrap`。旧版无绑定状态迁移时使用一次
`--allow-state-rebind`，随后必须检查新状态和执行报告。

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
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=quant_db
DB_USER=quant
DB_PASSWORD=你的数据库密码
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

同步器默认读取数据库 high-water mark，只重叠刷新每个标的最后一个完整 UTC 日；
需要重新审计全历史时才传 `--full-refresh`。

## Dry Run

首次小资金测试先按 `docs/TESTNET_START_CN.md` 执行，不要直接启动 daemon。

```bash
python scripts/binance_futures_testnet_executor.py \
  --config configs/backtest_v1.json \
  --exchange-leverage 2 \
  --target-gross-cap 1.25 \
  --hard-symbol-gross-limit 1.50 \
  --hard-account-gross-limit 1.50
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
  --exchange-leverage 2 \
  --target-gross-cap 1.25 \
  --hard-symbol-gross-limit 1.50 \
  --hard-account-gross-limit 1.50 \
  --execute
```

默认最多部署 `1000` USDT，单笔订单上限 `250` USDT，并保留 25% 可用保证金。
任何加仓还要求现有持仓的强平距离至少为 30%。测试初期应显式传更小的
`--max-deploy-usdt` 和 `--max-order-usdt`。

订单使用按信号日生成的确定性 client order ID，并要求 MARKET 订单确认为
`FILLED`。同一信号日重复运行会先查询已有订单，避免超时后重复下单。

## Daemon

### 一键 tmux 部署

以后更新和重启只运行：

```bash
bash scripts/vps_tmux_deploy.sh
```

脚本会停止旧daemon/monitor、备份runtime、执行`git pull --ff-only`、安装依赖、运行测试，
然后在两个tmux session中启动策略和监控。策略使用整个账户权益（`--max-deploy-usdt 0`），
单笔订单默认上限250 USDT。

可选覆盖：

```bash
RUN_AT_UTC=01:10 MONITOR_PORT=8765 MAX_ORDER_USDT=250 bash scripts/vps_tmux_deploy.sh
```

下面是手工启动方式，仅用于排查。

```bash
tmux new -s futures_daemon
source .venv/bin/activate
python scripts/run_daemon.py \
  --run-at-utc 01:10 \
  --run-on-start \
  --execute \
  --exchange-leverage 2 \
  --target-gross-cap 1.25
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
python scripts/serve_monitor.py --host 0.0.0.0 --port 8765
```

在`.env`设置`MONITOR_USERNAME`和`MONITOR_PASSWORD`，然后直接访问：

```text
http://VPS公网IP:8765/
```

防火墙放行端口：

```bash
sudo ufw allow 8765/tcp
```

这是最简单的Testnet监控方式。HTTP Basic Auth不加密传输，不要复用其他系统密码；正式资金部署
应升级为HTTPS反向代理。

## Backtest Review

```bash
python scripts/render_strategy_review_chart.py
```

默认完整窗口为 `2020-01-01` 到 `2026-05-18`，输出到：

```text
results/strategy_review/official_v1_full_20200101_20260518/
```

旧版报告包含 executor 未实现的 OHLC 日内阶梯成交，不再作为 release 指标。
当前基线已关闭该逻辑；打 tag 前必须重新生成并审查 metrics 与 release manifest。

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
