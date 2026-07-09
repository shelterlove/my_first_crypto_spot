# V4.7 Execution Experiment Review - 2026-07-08

本文记录 `v4_7_clean` 之后围绕执行层、outer layer、暴跌保护和杠杆设计做过的主要实验。目的不是保留所有实验分支，而是把有效结论沉淀下来，避免后续币种参数化时把失败补丁一起参数化。

## 当前保留候选

代码注册表目前只保留少量后续仍有价值的 V4.7 候选：

| strategy | 定位 | 结论 |
| --- | --- | --- |
| `v4_7_clean_event_exec_drift10_v1` | 交易次数修正后的基准 | 解决高频 target tracking，作为后续对照 |
| `v4_7_clean_event_exec_drift10_outer_deep_v1` | outer 少做多看对照 | 独立信号更干净，但错过 BNB 浅层机会 |
| `v4_7_clean_event_exec_drift10_outer_relaxed_v1` | outer 放宽对照 | 放宽未提高组合收益，事件质量下降 |
| `v4_7_clean_event_exec_drift10_intraday_shock_ladder_v7` | aggressive 主线候选 | 暴跌阶梯 + 深跌加仓，收益/回撤改善明显，但 max gross 接近 2.84 |
| `v4_7_clean_event_exec_drift10_intraday_shock_ladder_v11` | 受控 gross 参考 | gross cap 2.60，收益接近 drift10，回撤明显改善 |
| `v4_7_clean_event_exec_drift10_intraday_shock_ladder_v12` | 更贪婪 trend 杠杆研究版 | trend 1.90，收益最高的可讨论候选，但回撤和融资成本上升 |

已从公开注册表清掉的实验包括：crash guard、intraday delever、shock ladder v1-v6/v8-v10/v13-v15、低仓位恐慌加仓、post-shock trend cooldown。

## 指标对照

全窗口：requested `2019-01-01` 到 `2026-05-22`，实际数据通常为 `2020-01-01` 到 `2026-05-18`。百分比来自 `metrics.csv`。

| variant | total return | annual | max drawdown | avg position | trades | financing | max gross | max debt/equity | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| drift10 | 13043.04% | 114.92% | -46.29% | 59.16% | 670 | 1191.28 | 2.3056 | 1.3056 | 当前交易次数基准 |
| outer deep | 10713.37% | 108.44% | -45.55% | 56.45% | 660 | 913.78 | 2.3056 | 1.3056 | 少做但收益损失明显 |
| outer relaxed | 12978.09% | 114.75% | -46.28% | 60.56% | 682 | 1297.12 | 2.3056 | 1.3056 | 放宽没有价值 |
| shock v1 | 12297.94% | 112.96% | -43.36% | 59.24% | 814 | 1125.76 | 2.3056 | 1.3056 | close restore 损耗大 |
| shock v2 | 12295.34% | 112.95% | -43.32% | 59.00% | 788 | 1110.46 | 2.3056 | 1.3056 | 去掉 close restore 后更干净 |
| shock v5 | 12268.98% | 112.88% | -42.96% | 58.97% | 814 | 1099.92 | 2.3056 | 1.3056 | min position 0.80 合理 |
| shock v6 | 10893.10% | 108.98% | -44.67% | 58.83% | 883 | 994.36 | 2.2969 | 1.2969 | 触发太宽，失败 |
| shock v7 | 14515.17% | 118.52% | -42.84% | 59.00% | 822 | 1232.71 | 2.8394 | 1.8394 | aggressive 主线候选 |
| shock v9 | 14352.64% | 118.14% | -49.03% | 58.95% | 846 | 1220.62 | 2.8394 | 1.8394 | 低仓位恐慌加仓失败 |
| shock v10 | 12776.12% | 114.23% | -42.62% | 58.98% | 819 | 1139.11 | 2.4139 | 1.4139 | cap 2.50 过保守 |
| shock v11 | 13049.43% | 114.93% | -42.66% | 58.99% | 819 | 1156.20 | 2.5084 | 1.5084 | 受控 gross 参考 |
| shock v12 | 16163.24% | 122.22% | -44.77% | 62.07% | 869 | 1723.60 | 3.1422 | 2.1422 | trend 1.90 有效但风险上升 |
| shock v13 | 17215.21% | 124.41% | -47.55% | 65.16% | 924 | 2269.17 | 2.9639 | 1.9639 | trend 2.05 过激 |
| shock v14 | 8682.55% | 101.75% | -41.75% | 57.83% | 816 | 715.32 | 2.8394 | 1.8394 | v7 + 2日 trend cooldown，收益损失过大 |
| shock v15 | 8916.27% | 102.58% | -42.84% | 60.67% | 858 | 920.26 | 3.1422 | 2.1422 | v12 + 2日 trend cooldown，失败 |

## 交易次数结论

原始 `v4_7_clean` 交易数过高，主要原因是执行层每天追踪 transformed target，价格变化导致目标仓位漂移后反复小额调仓。

修正方式：

- execution transform 改为 event-triggered。
- raw action、outer lot enter/exit 仍触发交易。
- 对纯 target drift 设置 `EXECUTION_TRANSFORM_EVENT_DRIFT_GAP = 0.10`。

结果：交易数从原始约 1567 降到 drift10 的 670，收益反而提高到 13043.04%。这个方向已确认为有效，后续不应恢复每日精确追踪。

## Outer Layer 结论

outer layer 的职责应保持窄：极低位置买入固定数量，高位卖出，少做多看。它不应该变成每日百分比再平衡，也不应该承担主策略 recovery 的全部职责。

已验证结论：

- `outer_deep` 更干净但收益下降，说明 BNB 浅层结构性回撤机会很重要。
- `outer_relaxed` 放宽后收益未提高，事件质量下降，费用和融资成本上升。
- 不建议继续全局调 outer 阈值。
- 后续若继续优化 outer，应做币种参数化：BNB 可更宽，BTC/ETH 更偏 deep-value。

## Intraday Shock Ladder 结论

设计目标不是抄底，而是避免高 gross 状态下吃满日内急跌，并在更深跌幅处恢复仓位。

v7 当前规则：

- 日内从 open 跌到 `-10%`：卖出一段，将仓位降到 `current_pct - 0.35`，但不低于 `0.80`。
- 跌到 `-15%`：恢复 `sell_10` 卖出的数量。
- 跌到 `-20%`：额外加 `0.20` 目标仓位。
- 跌到 `-25%`：额外加 `0.20`。
- 跌到 `-30%`：额外加 `0.30`。
- 单币阶梯目标上限 `2.70`。

关键发现：

- v7 显著提高收益，并把最大回撤从 drift10 的 `-46.29%` 改善到 `-42.84%`。
- v7 的风险来自极端日 max gross 约 `2.8394`，接近 3x long exposure。
- v10/v11 的 gross cap 证明 2.50 太保守，2.60 是较合理的风险控制参考。
- 独立低仓位恐慌加仓失败：v9 最大回撤恶化到 `-49.03%`。

## 杠杆结论

常态杠杆收益主要来自 `trend_confirmed`，尤其 BNB 和 ETH；不是来自 BTC。

已验证结论：

- v12 将 `trend_mult/trend_cap` 从 `1.75` 提到 `1.90`，收益提高到 16163.24%，但最大回撤加深到 `-44.77%`，融资成本升到 1723.60。
- v13 的 `2.05` 证明更高 trend 杠杆仍能提高收益，但最大回撤恶化到 `-47.55%`，不适合作为主线。
- 后续如果继续提高杠杆，不应全局抬高，而应币种参数化或状态参数化。

## 最大回撤归因

v7 的最大回撤：

- peak: `2021-05-11`
- trough: `2021-05-19`
- max drawdown: `-42.84%`
- recover: `2021-11-07`

主要来源：

| symbol | peak equity | trough equity | leg drawdown | contribution |
| --- | ---: | ---: | ---: | ---: |
| BNB/USDT | 56.99 | 30.56 | -46.4% | -8.81 |
| ETH/USDT | 45.06 | 26.78 | -40.6% | -6.09 |
| BTC/USDT | 5.54 | 4.15 | -25.0% | -0.46 |

结论：40%+ 回撤主要来自高位 trend leverage 下 BNB/ETH 连续暴跌，BTC 不是主因。shock ladder 能减少单日跌穿，但无法完全解决连续暴跌期间主策略信号滞后的问题。

post-shock trend cooldown 被测试后放弃：

- v7 中 `sell_10` 后 1-2 天出现 trend 补杠杆买入的案例有 53 次，问题不是单一事件。
- 但简单 2 日禁用 trend 杠杆会大量错过正常牛市急跌后的反弹。
- v14/v15 收益大幅下降，说明该补丁过粗，不应保留。

## 清理后的开发原则

1. 不再向 `strategy_legacy.py` 增加策略逻辑。
2. 不再注册中间试参版本，实验结果进入文档，代码只保留可继续作为基准或候选的版本。
3. 后续优先做币种参数化，而不是继续堆全局补丁。
4. 币种参数化优先顺序：
   - `trend_mult/trend_cap`：BNB/ETH/BTC 分开。
   - outer entry/exit：BNB 保留浅层机会，BTC/ETH 更谨慎。
   - shock ladder：BNB 可容忍更激进恢复，BTC/ETH 可降低深跌加仓力度。
   - gross cap：作为风险预算，不应只作为单个实验分支。

## 当前建议

默认讨论基准：

- 稳健对照：`v4_7_clean_event_exec_drift10_v1`
- aggressive 主线：`v4_7_clean_event_exec_drift10_intraday_shock_ladder_v7`
- 风险受控参考：`v4_7_clean_event_exec_drift10_intraday_shock_ladder_v11`
- 更高收益研究：`v4_7_clean_event_exec_drift10_intraday_shock_ladder_v12`

下一步建议先做币种参数化，不再继续全局杠杆补丁。
