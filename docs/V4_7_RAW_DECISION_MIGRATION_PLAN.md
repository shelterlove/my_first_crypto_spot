# V4.7 Clean Architecture Migration

本文档记录 `v4_7_clean` 的整理目标、模块边界和验收方式。目标不是调参数，而是把 V4.7 的关键交易链从旧实验继承链中剥离出来，形成可读、可审计、行为固定的基准。

## 固定基准

- 固定策略：`v4_7`
- clean 策略：`v4_7_clean`
- 固定结果目录：`results/strategy_review/v4_7_full_20190101_20260522`
- 最终验证目录：`results/strategy_review/v4_7_clean_full_migration_verify_full_20190101_20260522`

固定基准指标：

- total return: `12196.41%`
- annual return: `112.68%`
- max drawdown: `-46.21%`
- avg position: `59.45%`
- trade count: `1567`
- financing cost: `1147.6895`
- margin calls: `0`
- max gross position: `2.2487`
- max debt/equity: `1.2487`

验收原则：

- 不覆盖固定基准结果目录。
- `v4_7_clean` 必须与 `v4_7` 行为等价。
- 允许 `metrics.csv` 的 `strategy` 名称不同。
- 允许 `actions.csv` 的 `reason` 中策略名不同。
- 其余核心行为表应等价，或差异必须能明确解释。

## 当前入口

- 注册位置：`src/crypto_spot_v1/benchmark.py`
- clean 入口：`src/crypto_spot_v1/v47/strategy.py`
- 类名：`V47CleanStrategy`

`V47CleanStrategy` 仍继承 `strategy_legacy.V47Strategy`，用途仅限：

- 复用旧构造函数初始化的大量状态字典、诊断计数器和配置常量。
- 保持旧结果文件导出字段兼容。
- 避免一次性重写所有非交易核心辅助代码。

关键交易流程已经由 `src/crypto_spot_v1/v47/` 接管。当前 v47 目录内不再通过 `super()._xxx` 调旧交易方法；代码扫描只剩 `LegacyV47Strategy` 继承和 `super().__init__`。

## 模块边界

- `raw_decision.py`：`compute_actions` 的 raw 决策调度顺序。
- `market.py`：context、regime、trend risk、drawdown risk、confirmed state。
- `signals.py`：accumulation、starter、value recovery、trend continuation、distribution exhaustion、recovery quality。
- `episode.py`：risk cycle、sell legs、recovery legs、recovery credit open/decay/check/event。
- `recovery.py`：episode recovery plan、reentry plan、recovery target/max buy、base-led recovery。
- `bear_base.py`：base/main quantity、bear-base target/floor/exit、base ledger 初始化、defensive sell ledger。
- `target.py`：mature target、phase target、base+tactical target 组合。
- `sleeve.py`：main/base/recovery/protected candidate sleeve 和 primary sleeve 仲裁。
- `sizing.py`：最终 sizing 调度、buy/sell sizing、guard、cooldown、max buy、min notional。
- `floor.py`：opportunity floor、protected floor、lifecycle low-base、protected base exit。
- `credit.py`：recovery credit plan、release、main buy consumption、BTC deep overlay once guard。
- `action.py`：action 构造和 action reason。
- `events.py`：事件写入顺序、last buy/sell call、risk cycle、credit release、sleeve accounting 入口。
- `accounting.py`：sleeve accounting、source ledger、base lot events、base/main quantity post-fill 记录。
- `lifecycle.py`：lifecycle shadow row 和 source quantity 拆解。
- `execution_engine.py`：raw target 到 execution target，low/trend 杠杆变换，outer fixed-quantity lots。
- `position_book.py`：执行层持仓、融资成本、强平审计和 execution rebalance。
- `config.py` / `models.py`：v47 执行层配置和结构体。

## 关键流程

1. `compute_actions` 构建 context。
2. `market` 生成 regime 和风险状态。
3. `signals` 生成买卖信号。
4. `episode` 更新 risk cycle / recovery state。
5. `target` 生成 mature target 与 phase target。
6. `sleeve` 生成候选 sleeve 并选择 primary sleeve。
7. `sizing` 计算 side、quantity、setup、guard。
8. `action` 生成 raw action。
9. `events` 写入 action 相关状态、risk cycle、credit、accounting、lifecycle shadow。
10. `execution_engine` 在回测执行层把 raw target 合成为最终 execution target。
11. `position_book` 记录真实执行、融资成本和强平审计。

## 仍保留 Legacy 的原因

当前保留继承，不代表继续依赖旧交易逻辑。保留原因是工程成本和兼容性：

- 旧类初始化了大量历史实验遗留状态，短期全部重写风险高。
- 导出脚本和回测 attrs 仍依赖部分历史字段存在。
- 配置常量目前仍集中在旧类层级，后续可逐步迁入 `v47/config.py`。

后续重构如果继续推进，优先顺序是：

1. 把 V4.7 实际用到的配置常量迁到 `v47/config.py`。
2. 给 v47 自己实现完整 `__init__`，减少对旧类初始化的依赖。
3. 删除不再注册、不再回测、不再被文档引用的历史实验策略。

## 最终验收命令

语法检查：

```powershell
python -m py_compile src\crypto_spot_v1\v47\accounting.py src\crypto_spot_v1\v47\action.py src\crypto_spot_v1\v47\bear_base.py src\crypto_spot_v1\v47\config.py src\crypto_spot_v1\v47\credit.py src\crypto_spot_v1\v47\episode.py src\crypto_spot_v1\v47\events.py src\crypto_spot_v1\v47\execution_engine.py src\crypto_spot_v1\v47\floor.py src\crypto_spot_v1\v47\lifecycle.py src\crypto_spot_v1\v47\market.py src\crypto_spot_v1\v47\models.py src\crypto_spot_v1\v47\position_book.py src\crypto_spot_v1\v47\raw_decision.py src\crypto_spot_v1\v47\recovery.py src\crypto_spot_v1\v47\signals.py src\crypto_spot_v1\v47\sizing.py src\crypto_spot_v1\v47\sleeve.py src\crypto_spot_v1\v47\strategy.py src\crypto_spot_v1\v47\target.py
```

最终回测：

```powershell
$env:DB_PORT='5433'
python scripts\render_strategy_review_chart.py --strategy v4_7_clean --start 2019-01-01 --end 2026-05-22 --output-dir results\strategy_review\v4_7_clean_full_migration_verify_full_20190101_20260522
```

对比基准：

```powershell
python scripts\compare_strategy_outputs.py results\strategy_review\v4_7_full_20190101_20260522 results\strategy_review\v4_7_clean_full_migration_verify_full_20190101_20260522
```

如果没有专用 compare 脚本，可用 pandas 对以下表逐表比较：

- `metrics.csv`
- `actions.csv`
- `decision_trace.csv`
- `candidate_orders.csv`
- `sleeve_events.csv`
- `sleeve_daily.csv`
- `base_lot_events.csv`
- `execution_transform_audit.csv`
- `outer_overlay_events.csv`
- `risk_cycles.csv`
- `defense_episodes.csv`
- `recovery_credit_events.csv`
- `recovery_credit_checks.csv`
- `lifecycle_state_shadow.csv`
- `diagnostics.csv`

## 后续策略优化边界

架构清理完成后，策略优化再回到三个问题：

- 低位暴露是否仍不足。
- 第一阶段恢复是否仍捕捉不足。
- 杠杆和 outer fixed lots 的风险约束是否需要统一。

在这些问题重新实验前，不应继续在 legacy 继承链上打补丁。新实验应基于 `v4_7_clean`，并使用独立 output dir。
