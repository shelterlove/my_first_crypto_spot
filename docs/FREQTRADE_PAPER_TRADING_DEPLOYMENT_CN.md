# Freqtrade 模拟盘部署说明

这份文档对应当前仓库里**推荐上线做模拟盘**的版本：

- 原生策略：`v3_4I`
- Freqtrade 壳策略：`CryptoSpotV34I`
- 交易所：`Binance Spot`
- 周期：`1d`
- 交易对：`BTC/USDT`、`ETH/USDT`、`BNB/USDT`

当前建议是：**先做模拟盘，不直接上真钱。**

---

## 1. 部署目标

你要在服务器上跑的是：

- 用 Freqtrade 接真实交易所行情
- 用 `CryptoSpotV34I` 产生实时信号
- 用 `dry_run: true` 做模拟下单
- 观察真实时间流里的信号、仓位变化、日志和执行稳定性

这不是历史回测，而是**实时模拟盘**。

---

## 2. 服务器准备

建议服务器环境：

- Linux 服务器
- Python `3.10+`
- 能联网访问交易所 API
- 能长期稳定运行

建议目录结构：

```text
/opt/crypto_spot_v1/
  src/
  scripts/
  configs/
  freqtrade_user_data/
  docs/
  pyproject.toml
```

如果你用 git 部署，直接把仓库拉到服务器即可。

---

## 3. 安装依赖

进入项目目录：

```bash
cd /opt/crypto_spot_v1
```

创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

升级 pip：

```bash
pip install --upgrade pip
```

安装项目和 Freqtrade 依赖：

```bash
pip install -e .[freqtrade]
```

如果你后面每次登录都要用这个环境：

```bash
source /opt/crypto_spot_v1/.venv/bin/activate
```

---

## 4. 检查代码和壳策略是否可用

先在服务器上验证：

```bash
python scripts/check_freqtrade_adapter.py
```

如果正常，会打印一个 `TargetPositionDecision(...)`，里面应能看到：

- `strategy` 对应 `v3_4I`
- 有 `buy / sell / hold` 之一

再检查 Freqtrade 是否能看到策略：

```bash
freqtrade list-strategies --userdir freqtrade_user_data
```

你应该能看到：

- `CryptoSpotV34I`

再检查配置里的交易对白名单：

```bash
freqtrade test-pairlist --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.example.json
```

---

## 5. 配置交易所 API

当前示例配置文件在：

[freqtrade_user_data/config/config.dryrun.example.json](D:/MyApp/crypto_spot_v1/freqtrade_user_data/config/config.dryrun.example.json)

不要直接改 example，建议复制一份：

```bash
cp freqtrade_user_data/config/config.dryrun.example.json freqtrade_user_data/config/config.dryrun.json
```

然后编辑：

```bash
nano freqtrade_user_data/config/config.dryrun.json
```

重点修改这段：

```json
"exchange": {
  "name": "binance",
  "key": "你的API_KEY",
  "secret": "你的API_SECRET",
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

### 建议

- 模拟盘建议使用**单独的 API key**
- 不要开提现权限
- 如果交易所支持，权限尽量最小
- 即使是 dry-run，也建议不要和正式交易 API 混用

### 一定确认

配置里必须是：

```json
"dry_run": true
```

不要误改成 `false`。

---

## 6. 下载历史数据

Freqtrade 需要本地行情数据。

运行：

```bash
freqtrade download-data \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.json \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT BNB/USDT \
  --timeframes 1d \
  --timerange 20170101-20260601 \
  --data-format-ohlcv feather \
  --prepend
```

下载后，数据一般会在：

```text
freqtrade_user_data/data/binance/
```

---

## 7. 启动模拟盘

启动命令：

```bash
freqtrade trade \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.json \
  --strategy CryptoSpotV34I
```

这就是正式的模拟盘启动方式。

---

## 8. 推荐的后台运行方式

### 方式 A：先用 `tmux`

安装：

```bash
sudo apt-get update
sudo apt-get install -y tmux
```

启动会话：

```bash
tmux new -s crypto_paper
```

在 tmux 里启动策略：

```bash
cd /opt/crypto_spot_v1
source .venv/bin/activate
freqtrade trade \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.json \
  --strategy CryptoSpotV34I
```

退出但不停止：

- 按 `Ctrl+B`
- 再按 `D`

重新进入：

```bash
tmux attach -t crypto_paper
```

### 方式 B：后面再改成 systemd

建议先用 `tmux` 跑通，再考虑 systemd 托管。

---

## 9. 启动后要检查什么

模拟盘上线后，不是只看程序有没有跑，而是重点看：

### 1）策略是否真的在出信号

观察日志里是否出现：

- `enter_long`
- `adjust_trade_position`
- `exit_long`
- 或 native signal 相关 reason

### 2）交易对是否正常

确认：

- BTC/USDT 有行为
- ETH/USDT 有行为
- BNB/USDT 有行为

如果只有个别币长期没动作，要看是不是策略本身没信号，还是数据/配置有问题。

### 3）仓位行为是否符合预期

重点观察：

- 是否频繁进出
- 是否有异常反复减仓
- 是否 stake_amount 和实际下单金额严重不符

### 4）reason 是否可解释

因为现在的核心价值之一就是：

- 每笔交易应该有明确的 `native_reason`

如果实盘日志里 reason 混乱或缺失，就不适合继续往真钱走。

---

## 10. 推荐每周检查项

建议每周至少检查一次：

- 本周每个币发生了哪些交易
- 每笔交易 reason 是什么
- 有无异常的频繁部分加仓/减仓
- 和回测对这个阶段的行为预期差多少
- bot 是否有重启、网络错误、钱包同步异常

必要时可以手动跑一次信号快照：

```bash
python scripts/generate_daily_signal.py --strategy v3_4I --output-dir results/daily_signals_server
```

---

## 11. 什么时候应该停止模拟盘并排查

出现以下情况建议先停下来排查：

- 日志里大量异常报错
- 信号频率明显高于回测预期
- 同一个币频繁买卖、行为很不合理
- 下单金额明显不对
- API 连接不稳定
- 账户/钱包状态和策略状态经常不同步

---

## 12. 升级和回滚方法

### 升级前

先停 bot：

- `Ctrl+C`
- 或断开 tmux 前先进入会话停止

然后保存当前版本：

```bash
git status
git rev-parse HEAD
```

最好打一个 tag 或记下 commit。

### 升级后

重新执行：

```bash
python scripts/check_freqtrade_adapter.py
freqtrade list-strategies --userdir freqtrade_user_data
```

确认没问题再重启 bot。

### 回滚

如果新版本有问题：

```bash
git checkout <上一个稳定commit>
```

然后重新：

```bash
python scripts/check_freqtrade_adapter.py
freqtrade trade --userdir freqtrade_user_data --config freqtrade_user_data/config/config.dryrun.json --strategy CryptoSpotV34I
```

---

## 13. 当前不建议上线的版本

不要用这些版本做主模拟盘：

- `v3_5D`
- `v3_5E`
- `v3_5F`
- `v3_5G`
- `v3_5H`

原因很简单：

- 它们是研究线上的 MIXED 回补执行实验
- 行为解释更细，但没有稳定超过 `v3_4I`

当前模拟盘推荐版本只有：

- `v3_4I`

---

## 14. 最短启动清单

如果你想最快跑起来，按这个顺序：

### 第一步：安装

```bash
cd /opt/crypto_spot_v1
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[freqtrade]
```

### 第二步：准备配置

```bash
cp freqtrade_user_data/config/config.dryrun.example.json freqtrade_user_data/config/config.dryrun.json
```

填写 API key / secret，并确认：

```json
"dry_run": true
```

### 第三步：检查

```bash
python scripts/check_freqtrade_adapter.py
freqtrade list-strategies --userdir freqtrade_user_data
```

### 第四步：下载数据

```bash
freqtrade download-data \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.json \
  --exchange binance \
  --pairs BTC/USDT ETH/USDT BNB/USDT \
  --timeframes 1d \
  --timerange 20170101-20260601 \
  --data-format-ohlcv feather \
  --prepend
```

### 第五步：启动

```bash
freqtrade trade \
  --userdir freqtrade_user_data \
  --config freqtrade_user_data/config/config.dryrun.json \
  --strategy CryptoSpotV34I
```

---

## 15. 最后建议

当前最合理的路线是：

- 先用 `CryptoSpotV34I` 跑 2-4 周模拟盘
- 不要一上来真钱
- 模拟盘稳定后，再决定要不要做下一步实盘验证

如果你下一步要，我可以继续帮你补：

- `config.dryrun.json` 的中文注释版模板
- `systemd` 服务文件
- 模拟盘运行日志检查清单
- 服务器首次部署 checklist
