# v3_16A 模拟盘部署文档

本文档用于把当前推荐的模拟盘候选 `v3_16A` 部署到 Freqtrade dry-run 环境。

## 目标版本

- 原生策略：`v3_16A`
- Freqtrade 壳策略：`CryptoSpotV316A`
- 配置模板：`freqtrade_user_data/config/config.dryrun.v316a.example.json`
- 交易所模型：Binance Spot
- 周期：`1d`
- 交易对：`BTC/USDT`、`ETH/USDT`、`BNB/USDT`
- 运行模式：只跑模拟盘，必须保持 `dry_run: true`

## 1. 部署原则

第一版目标不是直接实盘赚钱，而是验证真实时间流里的稳定性：

- Freqtrade 能否稳定加载策略。
- 实时信号是否和原生策略语义一致。
- 仓位调整金额是否符合预期。
- 是否出现异常重复买卖、频繁 partial sell / rebuy。
- 重启、断线、数据更新后是否正常恢复。

模拟盘期间不再临时修改 `v3_16A` 逻辑。研究线可以继续推进新版本，但部署线必须冻结。

## 2. 服务器准备

建议环境：

- Linux 服务器或 VPS
- Python `3.10+`
- 能长期稳定联网
- 能访问 Binance API
- 使用独立虚拟环境运行

推荐目录：

```text
/opt/crypto_spot_v1/
  src/
  scripts/
  configs/
  docs/
  freqtrade_user_data/
  pyproject.toml
```

不需要部署：

- `results/`
- `__pycache__/`
- 本地临时缓存

### 如果服务器上已经部署过旧版 4I

建议不要在旧目录里原地覆盖。更稳妥的方式是另起一个全新目录部署 `v3_16A`，例如：

```text
/opt/crypto_spot_v1_v316a/
```

旧目录先保留，例如：

```text
/opt/crypto_spot_v1_v34i/
```

这样做的好处：

- 不会混用旧的 Freqtrade 配置、缓存、日志和策略文件。
- 新旧版本可以清楚区分。
- 如果 `v3_16A` 启动失败，可以快速回到旧目录排查。
- 不需要立刻删除旧部署，降低误删配置和日志的风险。

推荐迁移顺序：

1. 停止旧版 `CryptoSpotV34I` bot。
2. 把新代码部署到新目录。
3. 在新目录里创建新虚拟环境。
4. 使用 `config.dryrun.v316a.json` 作为新配置。
5. 确认 `freqtrade list-strategies` 能看到 `CryptoSpotV316A`。
6. 启动新版 dry-run。
7. 新版稳定运行一段时间后，再考虑删除旧目录。

不要在确认新版可用前直接删除旧目录。

## 3. 安装依赖

进入项目目录：

```bash
cd /opt/crypto_spot_v1
```

创建并启用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

安装项目和 Freqtrade：

```bash
pip install -e .[freqtrade]
```

确认 Freqtrade 可用：

```bash
freqtrade --version
```

## 4. 准备 dry-run 配置

复制配置模板：

```bash
cp freqtrade_user_data/config/config.dryrun.v316a.example.json \
   freqtrade_user_data/config/config.dryrun.v316a.json
```

编辑配置：

```bash
nano freqtrade_user_data/config/config.dryrun.v316a.json
```

重点确认：

```json
{
  "dry_run": true,
  "strategy": "CryptoSpotV316A",
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "max_open_trades": 3,
  "timeframe": "1d"
}
```

交易所配置示例：

```json
"exchange": {
  "name": "binance",
  "key": "你的 API_KEY",
  "secret": "你的 API_SECRET",
  "ccxt_config": {},
  "ccxt_async_config": {},
  "pair_whitelist": [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT"
  ],
  "pair_blacklist": []
}
```

API key 建议：

- 使用单独的模拟盘 API key。
- 不要开启提现权限。
- dry-run 虽然不会真实下单，也不要混用主账户高权限 key。
- 如果交易所支持 IP 白名单，建议开启。

## 5. 下载 Freqtrade 行情数据

Freqtrade dry-run 和 backtest 需要本地行情文件。

```bash
freqtrade download-data \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.v316a.json \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT BNB/USDT \
  --timeframes 1d \
  --timerange 20170101-20260601 \
  --data-format-ohlcv feather \
  --prepend
```

数据通常会写入：

```text
freqtrade_user_data/data/binance/
```

`ETH/USDT` 和 `BNB/USDT` 的策略信号需要读取本地 `BTC_USDT-1d.feather` 计算 BTC regime，因此 BTC 数据必须存在。

## 6. 部署前检查

从项目根目录运行：

```bash
python scripts/check_freqtrade_adapter.py
```

这条检查当前默认仍会验证 adapter 基础链路。然后检查 `v3_16A` 能否被原生策略构建：

```bash
python - <<'PY'
from crypto_spot_v1.benchmark import build_strategy
s = build_strategy("v3_16A", 100, 20, 0.001)
print(type(s).__name__, s.name)
PY
```

期望输出类似：

```text
V316AStrategy v3_16A
```

检查 Freqtrade 能否看到壳策略：

```bash
freqtrade list-strategies --userdir freqtrade_user_data
```

期望能看到：

```text
CryptoSpotV316A
```

检查交易对白名单：

```bash
freqtrade test-pairlist \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.v316a.json
```

## 7. 启动模拟盘

前台启动：

```bash
freqtrade trade \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.v316a.json \
  --strategy CryptoSpotV316A
```

推荐先前台跑一段时间，确认没有启动错误、数据错误、pairlist 错误。

## 8. 使用 tmux 后台运行

安装 tmux：

```bash
sudo apt-get update
sudo apt-get install -y tmux
```

创建会话：

```bash
tmux new -s crypto_paper
```

在 tmux 中启动：

```bash
cd /opt/crypto_spot_v1
source .venv/bin/activate
freqtrade trade \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.v316a.json \
  --strategy CryptoSpotV316A
```

退出但不停止：

```text
Ctrl+B，然后按 D
```

重新进入：

```bash
tmux attach -t crypto_paper
```

## 9. 日常巡检

建议每天或每周检查一次。

重点看：

- bot 是否持续运行。
- 是否只有 `BTC/USDT`、`ETH/USDT`、`BNB/USDT`。
- `enter_tag` / `exit_tag` / adjustment reason 是否包含 `v3_16A`。
- 是否有异常频繁买卖。
- 是否有刚卖出又很快买回的 churn。
- 单笔 stake 是否和预期仓位变化接近。
- 钱包模拟余额是否异常。

策略语义重点：

- `target-gap` 是仓位补齐，不是独立好买点。
- `target-reduce` 是仓位减仓，不是独立好卖点。
- `risk-reduce`、`trend-break`、`trailing-profit-take` 是更高优先级的风险/保护行为。

## 10. 离线生成每日信号

如果只想看当前原生策略信号：

```bash
python scripts/generate_daily_signal.py \
  --strategy v3_16A \
  --output-dir results/daily_signals_v316a
```

这不会启动 Freqtrade，只生成离线信号记录。

## 11. 停机和重启

前台运行时停止：

```text
Ctrl+C
```

tmux 中停止：

```bash
tmux attach -t crypto_paper
```

然后在 bot 界面按：

```text
Ctrl+C
```

重启前建议重新跑：

```bash
python scripts/check_freqtrade_adapter.py
freqtrade list-strategies --userdir freqtrade_user_data
freqtrade test-pairlist --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.v316a.json
```

## 12. 异常停机条件

出现以下情况，应停止模拟盘并排查：

- Freqtrade 无法稳定更新 1d K 线。
- BTC regime 数据缺失或 BTC feather 文件不存在。
- 频繁出现同一天重复调整。
- 非预期交易对进入白名单。
- `dry_run` 被误改成 `false`。
- `native_reason` 不是 `v3_16A`，或 reason 为空但仍有交易。
- 单笔 stake 明显偏离目标仓位调整。
- 连续出现卖出后很快买回的循环。

## 13. 升级和回滚

升级策略前：

```bash
git status
git rev-parse HEAD
```

记录当前 commit 后再升级。

升级流程：

1. 停止 bot。
2. 拉取或部署新代码。
3. 重新安装依赖：`pip install -e .[freqtrade]`
4. 重新跑部署前检查。
5. 再启动 dry-run。

回滚流程：

1. 停止 bot。
2. 切回上一个确认可用 commit。
3. 重新跑部署前检查。
4. 启动 dry-run。

不要在 bot 运行中直接改策略代码。

## 14. 旧 4I 目录清理

如果新版 `v3_16A` 已经稳定运行，并且确认不需要旧日志和旧配置，可以清理旧目录。

清理前先确认没有旧 bot 在运行：

```bash
ps aux | grep freqtrade
```

如果旧版是 tmux 运行，查看会话：

```bash
tmux ls
```

如果还能看到旧会话，先进入并停止：

```bash
tmux attach -t crypto_paper
```

然后按：

```text
Ctrl+C
```

建议先改名备份，而不是直接删除：

```bash
mv /opt/crypto_spot_v1 /opt/crypto_spot_v1_v34i_backup
```

确认新版目录运行正常、旧目录确实不再需要后，再删除备份：

```bash
rm -rf /opt/crypto_spot_v1_v34i_backup
```

删除前一定确认路径，不要在不确定当前目录时运行 `rm -rf`。

## 15. 实盘前要求

`v3_16A` 当前只建议模拟盘，不建议直接实盘。

讨论小资金实盘前至少需要：

- dry-run 连续运行 2-4 周。
- 无异常 signal churn。
- 无异常 stake sizing。
- 重启和断线恢复测试通过。
- 真实时间信号和离线信号能解释一致。
- 明确最大投入、最大单笔、最大回撤停机规则。

## 16. 最短命令清单

```bash
cd /opt/crypto_spot_v1
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[freqtrade]

cp freqtrade_user_data/config/config.dryrun.v316a.example.json \
   freqtrade_user_data/config/config.dryrun.v316a.json

python scripts/check_freqtrade_adapter.py
freqtrade list-strategies --userdir freqtrade_user_data
freqtrade test-pairlist \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.v316a.json

freqtrade download-data \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.v316a.json \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT BNB/USDT \
  --timeframes 1d \
  --timerange 20170101-20260601 \
  --data-format-ohlcv feather \
  --prepend

freqtrade trade \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.v316a.json \
  --strategy CryptoSpotV316A
```
