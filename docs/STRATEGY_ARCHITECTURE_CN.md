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

1. `scripts/sync_binance_klines.py` 同步 USD-M futures 1d K 线；完整月份用 Binance Vision monthly archive，当前月回退到 daily archive，归档发布延迟的尾部再由 `/fapi/v1/klines` 补齐，且只接收已完成 UTC 日线。
2. K 线写入本地 PostgreSQL `candles` 表。
3. 回测、信号和执行器通过 `src/futures_v1/database.py` 读取数据。
4. 策略对 ETH、BNB 分别生成目标仓位和动作。
5. 执行器以最后一根已完成日线为信号，在下一开盘快照上计算目标，再读取 Binance testnet 账户、持仓和价格生成 MARKET 订单。
6. 执行器写出 JSON 报告，并更新 `runtime/futures_state.json` 虚拟 sleeve 状态。
7. monitor 读取 executor target snapshot、成交报告和 daemon 状态，生成静态 dashboard 数据。

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
- 状态文件缺失且持仓为零时初始化；持仓非零时拒绝隐式初始化。
- 状态文件损坏时拒绝继续，避免用错误账本下单。
- 状态绑定 API key 指纹、账户 alias 和配置 hash。
- 下单后重新读取账户权益并在同一周期完成账本对账。

相关测试在 `tests/test_futures_sleeve.py`。

## 风控边界

执行器层的硬约束：

- 默认 dry-run。
- 只有传 `--execute` 才会提交订单。
- 拒绝非 Binance USD-M Futures Testnet base URL。
- 拒绝超过 `--max-exchange-leverage` 的杠杆。
- 默认交易所杠杆 2x、策略软目标 gross 1.25x。
- 单 sleeve 和账户部署资金各自有 1.50x hard gross 上限。
- 默认部署资金上限 1000 USDT、单笔上限 250 USDT、可用保证金保留 25%。
- 有持仓的强平距离低于 30% 时禁止 BUY，但允许 reduce-only SELL。
- 检测到 short position 时退出。
- 只允许 ETH/USDT 和 BNB/USDT 两个交易标的。

策略层的主要风险控制：

- regime 风险过滤。
- 总 exposure 上限。
- 分层底仓和恢复买入限制。
- cooldown 约束。
- min notional 约束。
- episode 状态机避免过度重复交易。

## 历史研究基线

以下数字来自旧的 OHLC 日内阶梯成交口径，不再作为当前 executor 的可实现正式
基线：

- 窗口：`2020-01-01` 到 `2026-05-18`
- 策略年化：`135.93%`
- 策略总收益：`21588.90%`
- 策略最大回撤：`-43.39%`
- buy-and-hold 年化：`62.46%`
- buy-and-hold 总收益：`1992.99%`
- buy-and-hold 最大回撤：`-75.44%`

当前 deployable baseline 已关闭 executor 未实现的日内阶梯成交，并统一为已完成日线
信号在下一开盘执行。必须重新生成包含真实 futures fee、funding、slippage 和
liquidation stress 的 release manifest 后，才能固定新的正式业绩基线。

## 2026-07-10 工程一致性研究检查点

以下结果已关闭 OHLC-only 日内阶梯成交，并修正 BNB 上市前 sleeve 为现金，但仍使用
固定 borrow APR proxy，未包含真实 funding 和交易所强平模型，因此不是 release 指标：

- 组合年化：`128.80%`
- 组合总收益：`19489.29%`
- 组合最大回撤：`-51.88%`
- 日频 Sharpe：`1.54`
- 观察到的最大 gross：`2.299x`
- ETH：总收益 `13335.67%`，最大回撤 `-44.38%`
- BNB：总收益 `25642.91%`，最大回撤 `-58.99%`

风险研究结果：

| 研究项 | 年化 | 最大回撤 | 结论 |
|---|---:|---:|---|
| target cap 1.0 | 84.17% | -33.74% | 显著降低尾部风险 |
| target cap 1.25 | 101.52% | -39.88% | testnet 候选；历史 gross 峰值 1.360x |
| target cap 1.5 | 117.55% | -46.81% | 历史 gross 漂移到 1.789x，不作为默认目标 |
| cap 1.25 + 历史 funding | 89.19% | -41.89% | 日级 funding 近似，年化拖累约 7.94% |
| cap 1.25 + 2x funding 压力 | 79.26% | -43.79% | funding 压力后仍保持正收益 |
| target cap 2.0 | 128.66% | -51.88% | 收益接近饱和 |
| target cap 2.5 / 3.0 | 128.80% | -51.88% | 3.0 无可见历史增益 |
| ETH/BNB 75/25 | 122.75% | -45.93% | Sharpe 1.548，下一轮候选 |
| BTC neutral | 90.77% | -80.47% | BTC reference 有显著风险价值 |
| BTC delay 3d | 105.52% | -57.33% | 对额外延迟敏感 |

研究表由 `scripts/research_risk_caps.py`、`scripts/research_symbol_weights.py`、
`scripts/research_btc_regime.py` 和 `scripts/research_funding_stress.py` 生成。
测试网执行默认值已采用 2x exchange / 1.25x target / 1.50x hard gross；策略规则本身未重写。

执行层 2x 以上仓位实验见 `docs/EXECUTION_LEVERAGE_RESEARCH_CN.md`。trend cap 2.0 曾保留为
真实性研究候选，未修改正式策略或 testnet 风控参数。

路径依赖funding、滑点、部分成交、actual gross减仓、日内low强平近似及稳健性归因见
`docs/REALISM_ROBUSTNESS_RESEARCH_CN.md`。结果仍支持A作为正式基线，trend cap 2.0未晋升。

融资仓位门控研究见 `docs/FINANCING_GATE_RESEARCH_CN.md`。BTC regime二次门控无效；90% funding
分位门控仅保留为shadow候选，不影响当前testnet执行。

## 后续优化方向

优先级建议：

1. 给每类交易动作增加按年份、币种、regime 的归因表。
2. 检查最大回撤区间内的 sleeve 行为和风险门控是否过慢。
3. 做 ETH、BNB 分币种贡献拆解，确认是否有单一币种拖累。
4. 把已完成的 funding 日级近似升级为逐次结算、会影响后续仓位路径的回测模型。
5. 在保持代码简洁的前提下，继续减少重复规则和重复事件字段。
