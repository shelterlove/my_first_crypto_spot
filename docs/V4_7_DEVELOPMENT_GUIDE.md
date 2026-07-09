# V4.7 Strategy Development Guide

> 2026-07-08 update: execution-layer experiments after `v4_7_clean` are summarized in
> [V4_7_EXECUTION_EXPERIMENT_REVIEW_20260708.md](V4_7_EXECUTION_EXPERIMENT_REVIEW_20260708.md).
> The active post-clean candidate set is intentionally small: `drift10_v1`, `shock_ladder_v7`, `shock_ladder_v11`, and `shock_ladder_v12`.

本文档记录 `v4_7` 的策略结构、当前基准指标、审计文件和后续开发流程。目标是让后续开发先保护当前有效架构，再做可归因的独立实验。

## 当前状态

`v4_7` 是当前主线候选策略。它冻结自 `v4_6_outer_qty2x_v1`：

- 主策略：V4.6 clean execution-transform，即 `low26 + trend175` 杠杆执行层。
- 外层：固定数量 low-frequency outer lots，`OUTER_QTY_MULT = 2.0`。
- 主策略和外层分离：主策略先独立生成 raw 仓位，外层只在最终执行层叠加，不把外层仓位反馈给主策略决策。

代码入口：

- 活跃策略入口：[strategy_candidates.py](../src/crypto_spot_v1/strategy_candidates.py)
- 历史继承链：[strategy_legacy.py](../src/crypto_spot_v1/strategy_legacy.py)
- 策略注册：[benchmark.py](../src/crypto_spot_v1/benchmark.py)
- 回测执行：[backtest_event_driven.py](../src/crypto_spot_v1/backtest_event_driven.py)
- 结果导出：[render_strategy_review_chart.py](../scripts/render_strategy_review_chart.py)

当前主结果目录：

- `results\strategy_review\v4_7_full_20190101_20260522`

注意：命令请求窗口是 `2019-01-01` 到 `2026-05-22`，但当前结果实际数据覆盖通常是 `2020-01-01` 到 `2026-05-18`。指标解释以 `metrics.csv` 中的 `start` / `end` 为准。

## 基准指标

当前 `v4_7` 全窗口结果：

| 指标 | 数值 |
| --- | ---: |
| total return | 12196.41% |
| annual return | 112.68% |
| max drawdown | -46.21% |
| avg position | 59.45% |
| trade count | 1567 |
| financing cost | 1147.69 |
| margin calls | 0 |
| gross warnings | 107 |
| max gross position | 2.25x |
| max debt/equity | 1.25 |
| min margin buffer | 29.42 |

主要对照：

| 策略 | total return | annual | max drawdown | 说明 |
| --- | ---: | ---: | ---: | --- |
| `v4_2_exp_btc_tc_off_base_exit` | 2263.47% | 64.21% | -36.27% | 最好无杠杆 spot 基线 |
| `v4_2_exp_recovery_overlay_outer_qty2x_v1` | 3238.81% | 73.36% | -36.28% | 只加外层固定数量 2x |
| `v4_6` | 9231.45% | 103.68% | -45.66% | 只加主执行层杠杆 |
| `v4_7` | 12196.41% | 112.68% | -46.21% | 主执行层杠杆 + 外层固定数量 2x |
| `v4_6_trend2_outer_qty2x_v1` | 16497.86% | 122.93% | -51.25% | 激进对照，不是当前主线 |

`v4_6_trend2_outer_qty2x_v1` 收益最高，但融资成本、最大暴露和回撤明显增加，暂时只作为风险上沿对照，不作为干净基准。

## 策略分层

### 1. V4.2 raw 决策层

V4.2 决策层负责生成原始 spot 仓位，不直接知道外层 overlay，也不因为外层持仓改变内部状态。

核心流程：

1. 使用每个 symbol 的历史 candle 构建 context。
2. 根据价格位置、趋势、BTC 环境和风险特征生成 regime / risk / signals。
3. 计算目标仓位，包括 mature target、phase target 和 execution target。
4. 分别形成 main decision 和 base decision。
5. sizing 根据 side、quantity、setup、guard 生成 raw action。
6. risk cycle 记录 sell legs、recovery legs、未恢复预算。
7. sleeve accounting 将成交分配到 main / base，保持会计独立。

V4.7 继承的 V4.2 关键行为：

- BTC/USDT 的 `trend-cont` 买入关闭。
- ETH/USDT、BNB/USDT 的 `trend-cont` 保持开启。
- `bear-base-exit` 独立，只卖 base sleeve。
- 普通 distribution / risk-reduce 不应卖 base sleeve。
- main/base 会计独立，`exchange_quantity_after_est ~= main_quantity + base_quantity`。

### 2. 执行变换层

V4.7 使用 execution transform。回测中存在两个 portfolio：

- `decision_portfolio`：只执行 raw 策略动作，用于保持主策略决策链干净。
- `execution_portfolio`：执行最终变换后的真实交易，用于计算真实收益、融资、杠杆和风险。

执行层每日做法：

1. raw 策略在 `decision_portfolio` 上生成目标。
2. execution transform 读取 raw position pct。
3. 根据主杠杆规则和外层 fixed lot 规则计算最终目标仓位。
4. 根据 `execution_portfolio` 当前仓位和最终目标仓位生成真实 action。
5. 扣交易费、融资成本，记录 margin audit。

V4.7 的 clean execution settings：

| 参数 | 数值 | 说明 |
| --- | ---: | --- |
| `EXECUTION_TRANSFORM_BORROW_APR` | 0.10 | 年化融资成本 |
| `EXECUTION_TRANSFORM_MIN_TARGET_GAP` | 0.005 | 最小调仓带，低于 0.5pp 不交易 |
| `EXECUTION_TRANSFORM_MIN_NOTIONAL` | 5.0 | 最小交易金额 |
| `EXECUTION_TRANSFORM_MAINTENANCE_MARGIN` | 0.25 | 强平审计维护保证金 |
| `EXECUTION_TRANSFORM_WARNING_GROSS` | 1.85 | gross position 预警线 |

### 3. 主策略杠杆

V4.7 的主杠杆来自 `low26 + trend175`：

| 场景 | 条件摘要 | 变换 |
| --- | --- | --- |
| risk off | defense/distribution、DEFEND/EXIT/DISTRIBUTE、structural bear、risk >= 3、raw <= 5% 等 | 保持 raw |
| low recovery | low location + recovery active + risk <= 2 + raw >= 20% | `raw * 2.60`，上限 2.00x |
| trend confirmed | trend confirmed + BULL + risk <= 1 + raw >= 50% | `raw * 1.75`，上限 1.75x |
| normal | 不满足上面条件 | 保持 raw |

主杠杆只改变最终执行仓位，不改变 raw 决策层的状态和 action 归因。

### 4. 外层固定数量 lots

外层职责很窄：极低位置买入固定数量，极高位置卖出，尽量少交易。它不是恢复预算系统，也不是每日百分比再平衡系统。

外层目标大小：

| symbol | base outer target | V4.7 multiplier | entry overlay |
| --- | ---: | ---: | ---: |
| BTC/USDT | 14% | 2.0 | 28% |
| ETH/USDT | 14% | 2.0 | 28% |
| BNB/USDT | 10% | 2.0 | 20% |

固定数量含义：

- 买入时：`quantity = overlay_pct * execution_total / price`。
- 持有中：保存 quantity，不按最新权益重新平衡。
- 卖出时：卖出保存的 quantity。

这解决了早期 percent overlay 的问题：percent overlay 会因为权益变化反复调仓，容易变成伪高频；fixed quantity lot 更接近“低位买入后持有”。

外层低位入口：

- raw position >= 5%。
- 当前 symbol 至少 180 根历史数据。
- 如果 BTC regime 是 `BEAR` 且 `btc_roc_20 <= -12%`，不买。
- 价格足够低：满足任一条件：
  - 365 日位置 <= 0.15；
  - 365 日回撤 <= -60%；
  - 180 日回撤 <= -42%。
- 下跌有稳定迹象：满足任一条件：
  - 20 日低点反弹 >= 8%；
  - 5 日 ROC >= -4% 且 20 日 ROC >= -18%。

外层卖出：

- hard stop：相对入场后最低价再跌 30%，即 `price <= entry_low * 0.70`。
- high exit：至少持有 120 calls，盈利 >= 100%，且价格在高位区：
  - 365 日位置 >= 0.88 或 90 日 Donchian 位置 >= 0.90；
  - 20 日 ROC <= 12%，避免强趋势中太早卖。

## 关键不变量

开发时必须保护这些不变量：

- `v4_7` 结果目录不得被其他实验覆盖。
- 新实验必须使用新的 strategy name 和新的 output-dir。
- 主策略 raw 决策层不得读取外层持仓状态。
- 外层只在 execution layer 合并，不反向污染 `decision_portfolio`。
- BTC `trend-cont` off 不能误伤 ETH/BNB。
- 普通主仓卖出不得减少 `base_quantity`。
- `bear-base-exit` 不得减少 `main_quantity`。
- sleeve post-fill 口径必须满足：`exchange_quantity_after_est ~= main_quantity + base_quantity`。
- 杠杆实验必须检查融资成本、max gross、debt/equity、margin call。
- 对比行为时注意 strategy name 会改变 action reason；需要比较行为字段，不能只比较完整 action 行 hash。

## 标准输出文件

每次正式回测至少检查：

- `metrics.csv`：收益、年化、回撤、仓位、交易数、融资和杠杆风险。
- `actions.csv`：真实执行的买卖记录。
- `equity_curves.csv`：现金、持仓、总权益、仓位曲线。
- `execution_transform_audit.csv`：raw pct、transformed target、actual pct、融资、gross、margin。
- `outer_overlay_events.csv`：外层低频买卖事件。
- `risk_cycles.csv`：卖出预算、恢复 legs、未恢复预算。
- `sleeve_daily.csv`：main/base 数量和 exchange 数量差异。
- `sleeve_events.csv`：main/base 事件归因。
- `diagnostics.csv`：策略内部计数。
- `strategy_review.svg`：人工查看收益曲线和回撤路径。

## 标准命令

语法检查：

```powershell
python -m py_compile src\crypto_spot_v1\strategy_candidates.py src\crypto_spot_v1\benchmark.py src\crypto_spot_v1\backtest_event_driven.py scripts\render_strategy_review_chart.py
```

回测 V4.7：

```powershell
$env:DB_PORT='5433'
python scripts\render_strategy_review_chart.py --strategy v4_7 --start 2019-01-01 --end 2026-05-22 --output-dir results\strategy_review\v4_7_full_20190101_20260522
```

实验命名示例：

```powershell
$env:DB_PORT='5433'
python scripts\render_strategy_review_chart.py --strategy v4_7_exp_<feature>_v1 --start 2019-01-01 --end 2026-05-22 --output-dir results\strategy_review\v4_7_exp_<feature>_v1_full_20190101_20260522
```

## 开发流程

### 1. 先审计，不先调参

开始实验前先读：

- `metrics.csv`
- `actions.csv`
- `execution_transform_audit.csv`
- `outer_overlay_events.csv`
- `equity_curves.csv`

先回答三个问题：

- 问题发生在哪个币种、年份、行情段？
- 是 raw 决策层问题、主杠杆问题、外层 lot 问题，还是合并执行问题？
- 修改会不会破坏当前有效结构？

### 2. 新实验必须继承 V4.7

不要直接改 `V47Strategy`。新想法应新增类，并优先放在活跃入口模块或新的 V4.7 专用模块里。历史实验链保留在 `strategy_legacy.py`，不要继续往里面追加新实验。

```python
class V47ExpExampleV1Strategy(V47Strategy):
    @property
    def name(self) -> str:
        return "v4_7_exp_example_v1"
```

同时在 `benchmark.py` 注册新名字。每个实验使用独立 output-dir。

### 3. 一次只改变一个行为

可接受的实验粒度：

- 只改外层入口；
- 只改外层退出；
- 只改主杠杆条件；
- 只改融资/强平执行；
- 只改合并层冲突仲裁。

不要一次同时改入口、退出、杠杆倍数和风控，否则无法归因。

### 4. 验收指标

候选进入下一轮前至少满足：

- total/annual 相比 `v4_7` 有清晰改善，或明确降低风险。
- max drawdown 不应无解释地扩大。
- margin call 必须为 0，除非实验目标就是强平机制审计。
- max gross、debt/equity、financing cost 必须可解释。
- 外层事件数量保持低频，不能退化成高频调仓。
- 收益改善不能只来自一个孤立异常窗口。
- sleeve quantity diff 仍为浮点误差级。

### 5. 推荐对比集

任何 V4.7 后续实验至少对比：

- `v4_7`
- `v4_6`
- `v4_2_exp_btc_tc_off_base_exit`
- `v4_2_exp_recovery_overlay_outer_qty2x_v1`
- 当前实验策略

如果实验涉及趋势杠杆，还要对比：

- `v4_6_trend2_outer_qty2x_v1`

## 当前最重要的后续方向

优先方向不是继续堆参数，而是审计并稳定 V4.7 的执行风险：

1. 融资成本敏感性：确认 10% borrow APR 下有效，也要观察更高融资成本下是否仍有优势。
2. 实盘强平执行版：当前 margin call 为 0，但仍需保留 forced liquidation 对照。
3. 最大暴露集中期：重点审查 2023-03 ETH/BTC 和 2021-02 BNB。
4. 外层事件质量：确认低位买和高位卖的路径收益，不让外层变成频繁再平衡。
5. 最小调仓带：如果交易数过高，优先从 execution transform 的 min target gap 审计，而不是改 raw 策略。

暂不建议的方向：

- 直接把 trend-confirmed 杠杆从 1.75x 提到 2.0x 作为主线。
- 在 raw 决策层继续增加 recovery budget 补丁。
- 让外层仓位反向影响主策略状态。
- 做大规模币种参数优化；参数优化应等 V4.7 执行风险审计稳定后再做。

## 2026-07-07 outer layer 实验记录

基于 `v4_7_clean_event_exec_drift10_v1` 做过两组 outer low-entry 候选：

| 策略 | total return | annual | max drawdown | trade count | outer events | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `v4_7_clean_event_exec_drift10_v1` | 13043.04% | 114.92% | -46.29% | 670 | 18 | 当前对照 |
| `v4_7_clean_event_exec_drift10_outer_deep_v1` | 10713.37% | 108.44% | -45.55% | 660 | 10 | 更少做、更干净，但错过 BNB 浅层大机会，收益明显下降 |
| `v4_7_clean_event_exec_drift10_outer_relaxed_v1` | 12978.09% | 114.75% | -46.28% | 682 | 21 | 简单放宽未提升收益，费用、融资、hard stop 增加 |

归因结论：

- `outer_deep_v1` 的独立信号质量更好，bad-forward-days 从 136 降到 94，但完整组合收益下降，主要因为过滤了 BNB 2020/2021 等浅层回撤后的大机会。
- `outer_relaxed_v1` 增加了更多低位触发，但新增机会质量不足，带来更多费用、融资和 hard stop，组合收益略低于原 `drift10`。
- 当前不建议继续做全局 outer 阈值微调。后续如果重启 outer 优化，应优先考虑币种参数化：BNB 可保留浅层结构性回撤机会，BTC/ETH 更偏 deep-value、少做多看。
- 当前主线优先级转向单日大跌/熔断机制审计。
