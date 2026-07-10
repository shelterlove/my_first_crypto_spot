# Futures V1 策略说明

本文档说明当前正式版 `eth_bnb_futures_v1` 的策略口径和代码结构，方便代码审查、部署交接和后续优化。

## 策略口径

- 策略名：`eth_bnb_futures_v1`
- 交易标的：`ETH/USDT`、`BNB/USDT`
- 参考标的：`BTC/USDT`
- 周期：`1d`
- 市场：Binance USD-M Futures Testnet
- 方向：long-only
- 保证金：isolated
- 执行模式：默认 dry-run，传 `--execute` 后才提交 testnet 订单

BTC 不参与交易，只用于识别市场状态和过滤风险。ETH 和 BNB 分别独立计算目标仓位、交易动作和虚拟 sleeve 权益。

## 主入口

- 策略注册：`src/futures_v1/benchmark.py`
- 策略类：`src/futures_v1/strategy_core/strategy.py`
- 回测入口：`scripts/render_strategy_review_chart.py`
- 每日信号：`scripts/generate_daily_signal.py`
- testnet 执行器：`scripts/binance_futures_testnet_executor.py`
- daemon：`scripts/run_daemon.py`
- monitor 数据：`scripts/build_monitor_dashboard_data.py`
- monitor 服务：`scripts/serve_monitor.py`

## 数据流

1. `scripts/sync_binance_klines.py` 从 Binance Vision USD-M futures 数据源同步 1d K 线。
2. K 线写入本地 PostgreSQL `candles` 表。
3. 回测、信号和执行器通过 `src/futures_v1/database.py` 读取数据。
4. 策略对 ETH、BNB 分别生成目标仓位和动作。
5. 执行器读取 Binance testnet 账户、持仓和价格，按目标仓位生成 MARKET 订单。
6. 执行器写出 JSON 报告，并更新 `runtime/futures_state.json` 虚拟 sleeve 状态。
7. monitor 读取信号、执行报告和状态，生成静态 dashboard 数据。

## 策略模块

`src/futures_v1/strategy_core/strategy.py` 是组合壳，负责把各个规则模块串起来：

- `market.py`：构造当前市场上下文和 regime。
- `signals.py`：计算累积、恢复、趋势延续、分布衰竭等信号。
- `target.py`：生成目标仓位向量。
- `sleeve.py`：选择主 sleeve、恢复 sleeve、底仓 sleeve 等资金用途。
- `sizing.py`：把意图转换为买卖比例和数量。
- `action.py`：生成最终 buy/sell action。
- `events.py`：记录诊断、事件和审计轨迹。
- `accounting.py`：维护策略内部 sleeve 记账。
- `execution_engine.py`：把日线目标转换为执行层目标。

`src/futures_v1/*_rules.py` 保留一组策略规则 mixin，供 `StrategyCore` 组合使用。`strategy_core/*` 是更明确的 engine 层，承载主要业务逻辑。

## 决策流程

每个交易日、每个交易标的按以下顺序处理：

1. 构造 `StrategyContext`：当前价格、仓位、市值、历史指标、BTC 参考状态。
2. 构造 regime：根据趋势、均线、回撤和 BTC 状态判断市场环境。
3. 更新 episode：识别防守、恢复、分布等连续状态。
4. 构造 risk gate：判断是否需要限制买入、减仓或退出。
5. 构造 signals：计算恢复、趋势延续、底仓、分布衰竭等信号。
6. 构造 decision plan：决定当前主要意图。
7. 构造 target plan：给出目标仓位。
8. 构造 sleeve plan：决定资金来源和交易用途。
9. sizing：计算实际买卖比例和数量。
10. action：生成最终交易动作并记录审计信息。

## 虚拟 Sleeve 账本

执行器的虚拟 sleeve 文件是 `runtime/futures_state.json`。

核心原则：

- ETH 和 BNB 各自有独立 sleeve。
- 单币种交易盈亏只进入对应币种 sleeve。
- 充值、提现、手续费、资金费率等账户级差额平均分配到 ETH 和 BNB sleeve。
- 状态文件缺失时初始化。
- 状态文件损坏时拒绝继续，避免用错误账本下单。

相关测试在 `tests/test_futures_sleeve.py`。

## 风控边界

执行器层的硬约束：

- 默认 dry-run。
- 只有传 `--execute` 才会提交订单。
- 拒绝非 Binance USD-M Futures Testnet base URL。
- 拒绝超过 `--max-exchange-leverage` 的杠杆。
- `--target-gross-cap` 不得超过交易所杠杆。
- 检测到 short position 时退出。
- 只允许 ETH/USDT 和 BNB/USDT 两个交易标的。

策略层的主要风险控制：

- regime 风险过滤。
- 总 exposure 上限。
- 分层底仓和恢复买入限制。
- cooldown 约束。
- min notional 约束。
- episode 状态机避免过度重复交易。

## 正式版基线

当前正式版使用 futures K 线完整窗口回测：

- 窗口：`2020-01-01` 到 `2026-05-18`
- 策略年化：`135.93%`
- 策略总收益：`21588.90%`
- 策略最大回撤：`-43.39%`
- buy-and-hold 年化：`62.46%`
- buy-and-hold 总收益：`1992.99%`
- buy-and-hold 最大回撤：`-75.44%`

这份结果是正式版 v1 的参考基线。后续策略优化应保持独立分支或新实验目录，不应直接覆盖 v1 基线。

## 后续优化方向

优先级建议：

1. 给每类交易动作增加按年份、币种、regime 的归因表。
2. 检查最大回撤区间内的 sleeve 行为和风险门控是否过慢。
3. 做 ETH、BNB 分币种贡献拆解，确认是否有单一币种拖累。
4. 增加 futures 手续费和 funding 的压力测试。
5. 在保持代码简洁的前提下，继续减少重复规则和重复事件字段。
