from argparse import Namespace

from scripts.run_daemon import executor_command


def test_daemon_forwards_all_risk_boundaries() -> None:
    args = Namespace(
        config="configs/backtest_v1.json",
        exchange_leverage="2",
        target_gross_cap="1.25",
        hard_account_gross_limit="1.50",
        hard_symbol_gross_limit="1.50",
        max_deploy_usdt="1000",
        margin_buffer_fraction="0.25",
        min_liquidation_buffer="0.30",
        max_order_usdt="250",
        execute=True,
    )
    command = executor_command(args)
    joined = " ".join(command)
    for flag in (
        "--hard-account-gross-limit 1.50",
        "--hard-symbol-gross-limit 1.50",
        "--max-deploy-usdt 1000",
        "--margin-buffer-fraction 0.25",
        "--min-liquidation-buffer 0.30",
        "--max-order-usdt 250",
    ):
        assert flag in joined
    assert command[-1] == "--execute"
