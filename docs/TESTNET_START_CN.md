# Binance USD-M Futures Testnet 启动清单

## 当前候选风险参数

- 交易所杠杆：`2x` isolated，仅决定保证金结构。
- 策略 target gross cap：`1.25x`。
- 单 sleeve hard gross：`1.50x`。
- 整个部署 hard gross：`1.50x`。
- 可用保证金保留：`25%`。
- 加仓所需最低强平距离：`30%`。
- long-only、One-way Mode、ETHUSDT/BNBUSDT perpetual。

首次运行会把账户中未部署的 faucet 余额写入 state。以后部署权益等于账户权益减去这部分
未部署余额，因此策略盈亏不会被 `--max-deploy-usdt` 重置。不要在测试期间手工删除
`runtime/futures_state.json`，也不要把该 state 用于另一组 API key。

## 首轮人工测试

1. 在 `.env` 或进程环境设置 `BINANCE_FUTURES_TESTNET_API_KEY` 和
   `BINANCE_FUTURES_TESTNET_API_SECRET`。只允许 testnet key。
2. Binance USD-M Futures Testnet 切换为 One-way Mode，关闭 ETH/BNB 空头和其他合约持仓。
3. 先执行 dry-run：

```bash
python scripts/binance_futures_testnet_executor.py \
  --max-deploy-usdt 100 \
  --max-order-usdt 25
```

4. 审查最新 JSON：`base_url` 必须是 `demo-fapi.binance.com`，确认 target、BUY/SELL、
   projected account/symbol gross、部署权益和未部署余额。
5. 首次真实 testnet 下单：

```bash
python scripts/binance_futures_testnet_executor.py \
  --max-deploy-usdt 100 \
  --max-order-usdt 25 \
  --execute
```

6. 核对订单均为 `FILLED`，SELL 均为 `reduceOnly`，持仓为 isolated 2x，state 已更新，
   commission/funding income event 可审计，monitor 没有 danger alert。
7. 至少完成三个不同 UTC 日的人工周期后，再启用 daemon。不要在同一信号日通过删除 state
   或修改 client order ID 强制补单。

## Go / No-Go

以下任一条件成立都停止加仓：数据不是昨天的完整 UTC 日 K、出现 short/其他合约持仓、
state deployment/config 不匹配、订单未确认 `FILLED`、任一 hard gross 超限、强平距离低于
30%、账本总额与部署权益不一致或 daemon/monitor 报错。reduce-only 减仓仍可执行。
