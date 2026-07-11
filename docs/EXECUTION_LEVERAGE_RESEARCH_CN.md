# 执行层杠杆实验记录

## 2026-07-10：趋势与低位恢复系数拆分实验

### 目的

验证执行层是否应该允许 2x 以上仓位，并区分收益来自 `trend_confirmed` 还是
`low_recovery`。本轮只做研究，不修改正式策略默认值和 testnet 参数。

### 可重复运行

```bash
python scripts/research_execution_leverage.py
```

输出：

- `results/research/execution_leverage/summary.csv`
- `results/research/execution_leverage/calendar_returns.csv`
- `results/research/execution_leverage/manifest.json`

代码基准 commit：`cc486ef11f5cd8dcaef3e20230b55f11c335941f`。运行时工作区为 dirty，
所以结果必须结合 manifest 和本文件使用，不能仅凭 commit 重建。

### 统一假设

- 窗口：`2020-01-01` 至 `2026-05-18`
- 标的：ETH/USDT、BNB/USDT，各自独立 sleeve 后 50/50 合成
- BTC reference regime 保持不变
- next-open 日频执行
- research target gross cap：`2.40`
- OHLC intraday shock ladder：关闭
- 原回测的 fixed borrow APR proxy：保留
- 另做历史 funding 1x 和 2x 日级近似
- funding debit 不重新驱动后续仓位路径，因此 funding 指标是近似值

### 参数矩阵

| 变体 | low mult / cap | trend mult / cap | 目的 |
|---|---:|---:|---|
| A current | 2.60 / 2.00 | 1.75 / 1.75 | 当前基线 |
| B trend 2.0 | 2.60 / 2.00 | 2.00 / 2.00 | 只温和提高趋势仓位 |
| C trend 2.2, low reduced | 2.40 / 1.90 | 2.20 / 2.30 | 风险预算从逆势转向趋势 |
| D trend 2.3 | 2.60 / 2.00 | 2.30 / 2.30 | 保留低位规则，提高趋势上限 |
| E all 2.3 | 3.00 / 2.30 | 2.30 / 2.30 | 全面提高，作为压力对照 |

### 全窗口结果

| 变体 | 年化 | MDD | Sharpe | funding 1x 年化 / MDD | funding 2x 年化 / MDD | max gross |
|---|---:|---:|---:|---:|---:|---:|
| A | 128.80% | -51.88% | 1.541 | 113.58% / -54.75% | 101.60% / -57.29% | 2.299x |
| B | 138.60% | -57.44% | 1.503 | 120.92% / -60.79% | 107.25% / -63.64% | 2.762x |
| C | 147.28% | -61.87% | 1.487 | 128.16% / -65.54% | 113.38% / -68.58% | 3.336x |
| D | 151.01% | -63.74% | 1.478 | 130.65% / -67.65% | 115.04% / -70.82% | 3.667x |
| E | 150.96% | -63.74% | 1.478 | 130.58% / -67.65% | 114.94% / -70.82% | 3.667x |

Gross 尾部：

| 变体 | p95 gross | p99 gross | symbol-days gross > 2x |
|---|---:|---:|---:|
| A | 1.726x | 1.911x | 26 |
| B | 1.984x | 2.178x | 204 |
| C | 2.182x | 2.373x | 509 |
| D | 2.275x | 2.447x | 613 |
| E | 2.275x | 2.447x | 614 |

### 逐年收益

| 年份 | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| 2020 | 191.93% | 211.93% | 244.24% | 256.80% | 256.80% |
| 2021 | 1216.27% | 1381.28% | 1442.63% | 1513.78% | 1513.78% |
| 2022 | -3.99% | -4.00% | -4.00% | -4.01% | -4.01% |
| 2023 | 38.33% | 39.39% | 36.61% | 35.97% | 35.75% |
| 2024 | 150.51% | 174.15% | 199.94% | 204.54% | 204.65% |
| 2025 | 51.95% | 49.03% | 51.29% | 51.67% | 51.67% |
| 2026 YTD | -2.48% | -2.50% | -2.37% | -2.41% | -2.41% |

### 结论

1. 提高趋势系数确实增加历史收益，但 Sharpe 和收益/回撤效率持续下降。
2. D 与 E 收益几乎相同，说明提高 `low_recovery` 没有有效增益；额外收益主要来自趋势段。
3. `target gross cap=2.40` 只限制再平衡目标，不能限制两次日频执行之间因权益分母下降造成的
   actual gross 漂移。D/E 峰值 3.667x，不适合直接映射到 3x isolated 实盘。
4. B 是唯一保留的下一轮候选：funding 后年化比 A 高约 7.34 个百分点，但 MDD 恶化约
   6.04 个百分点，且 max gross 达到 2.762x。
5. C/D/E 暂时淘汰。没有实际 gross 强制减仓、真实逐次 funding 和交易所强平模拟前，不能部署。

### 开发发现

第一次实验曾错误地通过旧的 `StrategyConstants` 类属性注入参数，所有变体结果完全相同。
原因是重构后的 `ExecutionEngine` 在策略初始化时持有冻结的 `ExecutionConfig`。研究入口现已改为
替换 `strategy._core_config.execution` 并重建 engine，并有单元测试验证。以后执行参数实验必须走
`apply_execution_overrides()`，不能再直接 `setattr(strategy, EXEC_LOW_AWARE_...)`。

### 下一轮

只继续研究 A 与 B，并补以下约束后再判断：

1. actual symbol gross 达到 2.20x 时强制日频 reduce-only 回到 2.00x。
2. 对比 exchange leverage 3x 下的逐日强平距离，不使用固定 maintenance proxy 代替。
3. funding 逐次扣款并影响后续 sizing。
4. 分别报告 ETH、BNB 的最大回撤和 gross 峰值。
5. 测试 `ETH/BNB=65/35、75/25` 是否能用更低 BNB 风险换取趋势杠杆空间。

在这些实验完成前，正式 testnet 仍保持 `2x exchange / 1.25x target / 1.50x hard gross`。

后续真实性与稳健性实验已完成，见 `docs/REALISM_ROBUSTNESS_RESEARCH_CN.md`。加入路径依赖funding、
滑点、日内low强平近似和BTC延迟后，A仍保持最高Sharpe；B未晋升。
