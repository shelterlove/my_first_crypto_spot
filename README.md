# Crypto Spot V1

加密货币现货多头策略回测框架。核心策略是基于 EMA 状态机的仓位管理系统，在 `BULL`、`MIXED`、`BEAR` 三种状态间切换，并根据风险分数动态调整目标仓位。

当前稳定基准策略是 `v2_4`，实现类为 `V24Strategy`。当前实验候选沿用版本序列，放在 `v2_5` 及之后。

## 快速运行

```powershell
python scripts\run_v1_4_remaining.py
```

查看最新结果：

```text
results/v1_eval_upgrade/<timestamp>_v2_4/
```

关键文件：

- `model_review.md`             — 模型评审入口，先看这个
- `RESULTS_INDEX.md`            — 结果文件阅读顺序
- `summary_metrics.csv`
- `benchmark_metrics.csv`
- `regime_performance_report.csv`
- `strategy_optimization_comparison.csv`
- `diagnostics/diagnostic_quality_report.csv`
- `html_report.html`

## 常用命令

```powershell
python tests\test_v1_smoke.py
python scripts\run_v1_backtest.py --candidate v2_4
python -m compileall -q src scripts tests
```

## 核心目录

```text
configs/
  backtest_v1.json              # 回测配置

scripts/
  run_v1_4_remaining.py         # 当前评估入口
  run_v1_backtest.py            # 通用候选策略回测入口

src/crypto_spot_v1/
  strategy.py                   # V1SpotStrategy 基类
  strategy_candidates.py        # 候选策略继承链，当前基准为 V24Strategy
  strategy_utils.py             # EMA/ATR/Donchian/ROC 和市场状态检测
  benchmark.py                  # 策略注册和 V1BenchmarkRunner
  rolling_windows.py            # 滚动窗口回测
  metrics.py                    # 评分与绩效指标
  evaluation.py                 # 结果保存、审计、诊断和报告输出
  diagnostics.py                # per-bar diagnostics
  html_report.py                # HTML 报告
```

## 当前策略注册

可通过 `benchmark.py` 的 `STRATEGY_CLASSES` 使用：

- `v1`
- `v1_less_churn`
- `v1_1`
- `v1_2`
- `v1_3`
- `v1_7`
- `v1_9`
- `v1_9_orig`
- `v1_9A`
- `v1_9F`
- `v1_9H`
- `v1_9K`
- `v1_10`
- `v2_4`

## 策略继承链

```text
V1SpotStrategy
└── V1LessChurnStrategy
    └── V11Strategy
        └── V12Strategy
            └── V13Strategy
                └── V17Strategy
                    └── V19Strategy
                        └── V19AStrategy
                            └── V19FStrategy
                                └── V19HStrategy
                                    └── V19KStrategy
                                        └── V110Strategy
                                            └── V24Strategy (v2_4)
                                                └── V25Strategy (v2_5)
```

## 最近完整评估

本仓库当前已完整跑过：

```text
results/v1_eval_upgrade/20260528_135501_v2_4
```

核心结果：

- score: `0.6500`
- windows: `165`
- median excess vs Buy & Hold: `0.0847`
- win rate vs Buy & Hold: `0.5394`
- mean max drawdown: `-0.3768`
- bull excess vs Buy & Hold: `0.3155`
- bear win rate vs Buy & Hold: `0.9444`
- diagnostics quality: all checks passed

当前结论：`v2_4` 明显优于 `v1_less_churn`，但 bull win rate 仍低于 0.35，晋级条件尚未完全满足。

## 开发约定

- 不要改写 `v1` 基线逻辑，除非明确做迁移修复。
- 新策略优先继承当前最佳版本类或最接近的父类。
- 新策略必须在 `benchmark.py` 注册。
- 优化后至少运行 smoke test 和 compile check。
- 完整评估使用 `scripts/run_v1_4_remaining.py`。
- `.env`、本地虚拟环境和 `results/` 不提交。
