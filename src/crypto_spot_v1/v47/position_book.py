"""Execution-position helpers for the V4.7 transformed portfolio."""

from __future__ import annotations

import pandas as pd

from ..strategy_rebalance import Action, PortfolioState, PortfolioStrategyBase, PositionState


def execution_min_notional(strategy: PortfolioStrategyBase) -> float:
    base_min = float(getattr(strategy, "min_notional", 0.0) or 0.0)
    transform_min = float(getattr(strategy, "EXECUTION_TRANSFORM_MIN_NOTIONAL", base_min) or 0.0)
    return max(base_min, transform_min)


def execution_min_target_gap(strategy: PortfolioStrategyBase) -> float:
    return float(getattr(strategy, "EXECUTION_TRANSFORM_MIN_TARGET_GAP", 0.0) or 0.0)


def maintenance_margin(strategy: PortfolioStrategyBase) -> float:
    return float(getattr(strategy, "EXECUTION_TRANSFORM_MAINTENANCE_MARGIN", 0.25) or 0.0)


def warning_gross(strategy: PortfolioStrategyBase) -> float:
    return float(getattr(strategy, "EXECUTION_TRANSFORM_WARNING_GROSS", 1.85) or 0.0)


def borrow_apr(strategy: PortfolioStrategyBase) -> float:
    return float(getattr(strategy, "EXECUTION_TRANSFORM_BORROW_APR", 0.0) or 0.0)


def force_liquidation(strategy: PortfolioStrategyBase) -> bool:
    return bool(getattr(strategy, "EXECUTION_TRANSFORM_FORCE_LIQUIDATION", False))


def intraday_shock_ladder_enabled(strategy: PortfolioStrategyBase) -> bool:
    return bool(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_SHOCK_LADDER_V1", False))


def build_intraday_shock_actions(
    strategy: PortfolioStrategyBase,
    portfolio: PortfolioState,
    intraday_rows: dict[str, pd.Series],
    fee_rate: float,
) -> list[Action]:
    if not intraday_shock_ladder_enabled(strategy):
        return []
    return build_intraday_shock_ladder_actions(
        strategy=strategy,
        portfolio=portfolio,
        intraday_rows=intraday_rows,
        fee_rate=fee_rate,
    )


def build_intraday_shock_ladder_actions(
    strategy: PortfolioStrategyBase,
    portfolio: PortfolioState,
    intraday_rows: dict[str, pd.Series],
    fee_rate: float,
) -> list[Action]:
    sell_drop = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_SELL_DROP", -0.10))
    restore_drop = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_RESTORE_DROP", -0.15))
    add_drop = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_ADD_DROP", -0.20))
    min_position = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_MIN_POSITION", 1.20))
    floor_position = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_FLOOR_POSITION", 1.20))
    reduce_step = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_REDUCE_STEP", 0.35))
    add_step = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_ADD_STEP", 0.20))
    max_position = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_MAX_POSITION", 2.30))
    restore_close_enabled = bool(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_RESTORE_CLOSE", True))
    restore_close_below_sell = bool(getattr(
        strategy,
        "EXECUTION_TRANSFORM_INTRADAY_LADDER_RESTORE_CLOSE_BELOW_SELL",
        False,
    ))
    min_notional = execution_min_notional(strategy)
    gross_mark_prices = {
        symbol: float(row.get("close", 0.0) or 0.0)
        for symbol, row in intraday_rows.items()
    }
    actions: list[Action] = []

    for symbol, row in intraday_rows.items():
        open_price = float(row.get("open", 0.0) or 0.0)
        low_price = float(row.get("low", 0.0) or 0.0)
        close_price = float(row.get("close", 0.0) or 0.0)
        if open_price <= 0.0 or low_price <= 0.0 or close_price <= 0.0:
            continue
        intraday_drop = low_price / open_price - 1.0
        if intraday_drop > sell_drop:
            continue

        sell_price = open_price * (1.0 + sell_drop)
        restore_price = open_price * (1.0 + restore_drop) if intraday_drop <= restore_drop else close_price
        add_price = open_price * (1.0 + add_drop)
        if sell_price <= 0.0 or restore_price <= 0.0 or add_price <= 0.0:
            continue

        pos = portfolio.positions.get(symbol, PositionState())
        if pos.quantity <= 1e-12:
            continue
        total_value_at_sell = portfolio.cash + pos.quantity * sell_price
        if total_value_at_sell <= 0.0:
            continue
        current_pct = pos.quantity * sell_price / total_value_at_sell
        if current_pct <= min_position:
            continue

        target_after_sell = max(floor_position, current_pct - reduce_step)
        if target_after_sell >= current_pct:
            continue
        desired_qty_after_sell = max(0.0, target_after_sell * total_value_at_sell / sell_price)
        sell_qty = min(pos.quantity, max(0.0, pos.quantity - desired_qty_after_sell))
        sell_notional = sell_qty * sell_price
        if sell_qty <= 1e-12 or sell_notional < max(min_notional, 1e-9):
            continue

        sold_qty = sell_qty
        actions.append(Action(
            symbol=symbol,
            side="sell",
            quantity=sell_qty,
            price=sell_price,
            reason=(
                f"{strategy.name}_sell_intraday-shock-ladder"
                f"_low{intraday_drop:.0%}_from{current_pct:.0%}_to{target_after_sell:.0%}"
            ),
            diagnostics={
                "setup": "intraday-shock-ladder",
                "target_pct": target_after_sell,
                "actual_position_before": current_pct,
                "actual_step_pct": target_after_sell - current_pct,
                "execution_transform": True,
                "intraday_shock_ladder": True,
                "intraday_ladder_leg": "sell_10",
                "intraday_open": open_price,
                "intraday_low": low_price,
                "intraday_close": close_price,
                "intraday_drop": intraday_drop,
                "fee_rate": fee_rate,
            },
        ))

        if intraday_drop <= restore_drop:
            restore_reason = "restore_15"
        elif restore_close_enabled or (restore_close_below_sell and close_price <= sell_price):
            restore_reason = "restore_close_below_sell" if close_price <= sell_price else "restore_close"
        else:
            continue
        restore_qty = sold_qty
        restore_notional = restore_qty * restore_price
        if restore_qty > 1e-12 and restore_notional >= max(min_notional, 1e-9):
            actions.append(Action(
                symbol=symbol,
                side="buy",
                quantity=restore_qty,
                price=restore_price,
                reason=(
                    f"{strategy.name}_buy_intraday-shock-ladder-{restore_reason}"
                    f"_low{intraday_drop:.0%}"
                ),
                diagnostics={
                    "setup": "intraday-shock-ladder",
                    "target_pct": current_pct,
                    "execution_transform": True,
                    "intraday_shock_ladder": True,
                    "intraday_ladder_leg": restore_reason,
                    "intraday_open": open_price,
                    "intraday_low": low_price,
                    "intraday_close": close_price,
                    "intraday_drop": intraday_drop,
                    "fee_rate": fee_rate,
                },
            ))

        if intraday_drop > add_drop or add_step <= 0.0:
            continue
        total_value_at_add = portfolio.cash + pos.quantity * add_price
        if total_value_at_add <= 0.0:
            continue
        pct_at_add_after_restore = pos.quantity * add_price / total_value_at_add
        target_after_add = min(max_position, pct_at_add_after_restore + add_step)
        if target_after_add <= pct_at_add_after_restore:
            continue
        planned_buy_notional = 0.0
        desired_qty_after_add = target_after_add * total_value_at_add / add_price
        add_qty = max(0.0, desired_qty_after_add - pos.quantity)
        add_qty = _cap_buy_qty_by_gross(
            strategy=strategy,
            portfolio=portfolio,
            symbol=symbol,
            price=add_price,
            requested_qty=add_qty,
            current_prices=gross_mark_prices,
            reserved_notional=planned_buy_notional,
        )
        add_notional = add_qty * add_price
        if add_qty <= 1e-12 or add_notional < max(min_notional, 1e-9):
            continue
        actions.append(Action(
            symbol=symbol,
            side="buy",
            quantity=add_qty,
            price=add_price,
            reason=(
                f"{strategy.name}_buy_intraday-shock-ladder-add20"
                f"_low{intraday_drop:.0%}_to{target_after_add:.0%}"
            ),
            diagnostics={
                "setup": "intraday-shock-ladder",
                "target_pct": target_after_add,
                "actual_position_before": pct_at_add_after_restore,
                "actual_step_pct": target_after_add - pct_at_add_after_restore,
                "execution_transform": True,
                "intraday_shock_ladder": True,
                "intraday_ladder_leg": "add_20",
                "intraday_open": open_price,
                "intraday_low": low_price,
                "intraday_close": close_price,
                "intraday_drop": intraday_drop,
                "fee_rate": fee_rate,
            },
        ))
        planned_buy_notional += add_notional

        virtual_qty = pos.quantity + add_qty
        for tier_drop, tier_step, tier_label in getattr(
            strategy,
            "EXECUTION_TRANSFORM_INTRADAY_LADDER_EXTRA_ADD_TIERS",
            [],
        ):
            tier_drop = float(tier_drop)
            tier_step = float(tier_step)
            tier_label = str(tier_label)
            if intraday_drop > tier_drop or tier_step <= 0.0:
                continue
            tier_price = open_price * (1.0 + tier_drop)
            if tier_price <= 0.0:
                continue
            total_value_at_tier = portfolio.cash + virtual_qty * tier_price
            if total_value_at_tier <= 0.0:
                continue
            pct_at_tier = virtual_qty * tier_price / total_value_at_tier
            target_after_tier = min(max_position, pct_at_tier + tier_step)
            if target_after_tier <= pct_at_tier:
                continue
            desired_qty_after_tier = target_after_tier * total_value_at_tier / tier_price
            tier_qty = max(0.0, desired_qty_after_tier - virtual_qty)
            tier_qty = _cap_buy_qty_by_gross(
                strategy=strategy,
                portfolio=portfolio,
                symbol=symbol,
                price=tier_price,
                requested_qty=tier_qty,
                current_prices=gross_mark_prices,
                reserved_notional=planned_buy_notional,
            )
            tier_notional = tier_qty * tier_price
            if tier_qty <= 1e-12 or tier_notional < max(min_notional, 1e-9):
                continue
            virtual_qty += tier_qty
            planned_buy_notional += tier_notional
            actions.append(Action(
                symbol=symbol,
                side="buy",
                quantity=tier_qty,
                price=tier_price,
                reason=(
                    f"{strategy.name}_buy_intraday-shock-ladder-{tier_label}"
                    f"_low{intraday_drop:.0%}_to{target_after_tier:.0%}"
                ),
                diagnostics={
                    "setup": "intraday-shock-ladder",
                    "target_pct": target_after_tier,
                    "actual_position_before": pct_at_tier,
                    "actual_step_pct": target_after_tier - pct_at_tier,
                    "execution_transform": True,
                    "intraday_shock_ladder": True,
                    "intraday_ladder_leg": tier_label,
                    "intraday_open": open_price,
                    "intraday_low": low_price,
                    "intraday_close": close_price,
                    "intraday_drop": intraday_drop,
                    "fee_rate": fee_rate,
                },
            ))
    return actions


def _cap_buy_qty_by_gross(
    strategy: PortfolioStrategyBase,
    portfolio: PortfolioState,
    symbol: str,
    price: float,
    requested_qty: float,
    current_prices: dict[str, float],
    reserved_notional: float = 0.0,
) -> float:
    max_gross = float(getattr(strategy, "EXECUTION_TRANSFORM_INTRADAY_LADDER_MAX_GROSS", 0.0) or 0.0)
    if max_gross <= 0.0 or requested_qty <= 0.0 or price <= 0.0:
        return requested_qty
    total_value = portfolio.cash
    gross_value = 0.0
    for pos_symbol, pos_state in portfolio.positions.items():
        mark_price = price if pos_symbol == symbol else float(current_prices.get(pos_symbol, 0.0) or 0.0)
        if mark_price <= 0.0:
            continue
        value = pos_state.quantity * mark_price
        total_value += value
        gross_value += max(0.0, value)
    if total_value <= 0.0:
        return 0.0
    gross_room = max_gross * total_value - gross_value - max(0.0, reserved_notional)
    if gross_room <= 0.0:
        return 0.0
    return min(requested_qty, gross_room / price)


def transformed_target_for_symbol(
    strategy: PortfolioStrategyBase,
    symbol: str,
    raw_position_pct: float,
    candles_by_symbol: dict[str, pd.DataFrame],
    current_prices: dict[str, float],
    decision_portfolio: PortfolioState,
    execution_portfolio: PortfolioState,
) -> float:
    if hasattr(strategy, "execution_target_for_symbol_with_portfolios"):
        return float(strategy.execution_target_for_symbol_with_portfolios(
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
            decision_portfolio=decision_portfolio,
            execution_portfolio=execution_portfolio,
        ))
    return float(strategy.execution_target_for_symbol(
        symbol=symbol,
        raw_position_pct=raw_position_pct,
        candles_by_symbol=candles_by_symbol,
        current_prices=current_prices,
    ))


def transformed_target_diagnostics_for_symbol(
    strategy: PortfolioStrategyBase,
    symbol: str,
    raw_position_pct: float,
    candles_by_symbol: dict[str, pd.DataFrame],
    current_prices: dict[str, float],
    decision_portfolio: PortfolioState,
    execution_portfolio: PortfolioState,
) -> tuple[float, str]:
    if hasattr(strategy, "execution_transform_diagnostics_for_symbol_with_portfolios"):
        target_pct, transform_reason = strategy.execution_transform_diagnostics_for_symbol_with_portfolios(
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
            decision_portfolio=decision_portfolio,
            execution_portfolio=execution_portfolio,
        )
        return float(target_pct), str(transform_reason)
    if hasattr(strategy, "execution_transform_diagnostics_for_symbol"):
        target_pct, transform_reason = strategy.execution_transform_diagnostics_for_symbol(
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
        )
        return float(target_pct), str(transform_reason)
    if hasattr(strategy, "execution_target_for_symbol"):
        target_pct = transformed_target_for_symbol(
            strategy=strategy,
            symbol=symbol,
            raw_position_pct=raw_position_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=current_prices,
            decision_portfolio=decision_portfolio,
            execution_portfolio=execution_portfolio,
        )
        return float(target_pct), "raw"
    return float(raw_position_pct), "raw"


def build_rebalance_actions(
    strategy: PortfolioStrategyBase,
    decision_portfolio: PortfolioState,
    execution_portfolio: PortfolioState,
    candles_by_symbol: dict[str, pd.DataFrame],
    execution_prices: dict[str, float],
    fee_rate: float,
    raw_actions: list[Action] | None = None,
) -> list[Action]:
    if bool(getattr(strategy, "EXECUTION_TRANSFORM_EVENT_TRIGGERED", False)):
        return build_event_triggered_rebalance_actions(
            strategy=strategy,
            decision_portfolio=decision_portfolio,
            execution_portfolio=execution_portfolio,
            candles_by_symbol=candles_by_symbol,
            execution_prices=execution_prices,
            fee_rate=fee_rate,
            raw_actions=raw_actions or [],
        )

    actions: list[Action] = []
    min_notional = execution_min_notional(strategy)
    min_target_gap = execution_min_target_gap(strategy)
    for symbol, price in execution_prices.items():
        if price <= 0.0:
            continue
        decision_pos = decision_portfolio.positions.get(symbol, PositionState())
        execution_pos = execution_portfolio.positions.get(symbol, PositionState())
        decision_total = decision_portfolio.cash + decision_pos.quantity * price
        execution_total = execution_portfolio.cash + execution_pos.quantity * price
        if decision_total <= 0.0 or execution_total <= 0.0:
            continue
        raw_pct = decision_pos.quantity * price / decision_total
        current_pct = execution_pos.quantity * price / execution_total
        target_pct = max(0.0, transformed_target_for_symbol(
            strategy=strategy,
            symbol=symbol,
            raw_position_pct=raw_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=execution_prices,
            decision_portfolio=decision_portfolio,
            execution_portfolio=execution_portfolio,
        ))
        if abs(target_pct - current_pct) < min_target_gap:
            continue
        desired_qty = target_pct * execution_total / price
        delta_qty = desired_qty - execution_pos.quantity
        notional = abs(delta_qty) * price
        if notional < max(min_notional, 1e-9):
            continue
        side = "buy" if delta_qty > 0.0 else "sell"
        qty = abs(delta_qty)
        if side == "sell":
            qty = min(qty, execution_pos.quantity)
            if qty <= 1e-12:
                continue
        estimated_after_qty = execution_pos.quantity + qty if side == "buy" else execution_pos.quantity - qty
        estimated_after_pct = estimated_after_qty * price / execution_total if execution_total > 0.0 else 0.0
        actions.append(Action(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            reason=(
                f"{strategy.name}_{side}_execution-transform"
                f"_raw{raw_pct:.0%}_target{target_pct:.0%}"
            ),
            diagnostics={
                "setup": "execution-transform",
                "target_pct": target_pct,
                "raw_position_pct": raw_pct,
                "transformed_target_pct": target_pct,
                "actual_position_before": current_pct,
                "actual_position_after": estimated_after_pct,
                "actual_step_pct": estimated_after_pct - current_pct,
                "execution_transform": True,
                "fee_rate": fee_rate,
            },
        ))
    return actions


def build_event_triggered_rebalance_actions(
    strategy: PortfolioStrategyBase,
    decision_portfolio: PortfolioState,
    execution_portfolio: PortfolioState,
    candles_by_symbol: dict[str, pd.DataFrame],
    execution_prices: dict[str, float],
    fee_rate: float,
    raw_actions: list[Action],
) -> list[Action]:
    actions: list[Action] = []
    min_notional = execution_min_notional(strategy)
    raw_symbols = {action.symbol for action in raw_actions}
    outer_state = getattr(strategy, "_outer_overlay_state_by_symbol", {})

    for symbol, price in execution_prices.items():
        if price <= 0.0:
            continue
        decision_pos = decision_portfolio.positions.get(symbol, PositionState())
        execution_pos = execution_portfolio.positions.get(symbol, PositionState())
        decision_total = decision_portfolio.cash + decision_pos.quantity * price
        execution_total = execution_portfolio.cash + execution_pos.quantity * price
        if decision_total <= 0.0 or execution_total <= 0.0:
            continue

        raw_pct = decision_pos.quantity * price / decision_total
        current_pct = execution_pos.quantity * price / execution_total
        state_before = dict(outer_state.get(symbol, {}) or {})
        outer_qty_before = max(0.0, float(state_before.get("quantity", 0.0) or 0.0))
        target_pct, transform_reason = transformed_target_diagnostics_for_symbol(
            strategy=strategy,
            symbol=symbol,
            raw_position_pct=raw_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=execution_prices,
            decision_portfolio=decision_portfolio,
            execution_portfolio=execution_portfolio,
        )
        target_pct = max(0.0, float(target_pct))
        state_after = dict(outer_state.get(symbol, {}) or {})
        outer_qty_after = max(0.0, float(state_after.get("quantity", 0.0) or 0.0))

        if symbol in raw_symbols:
            desired_qty = target_pct * execution_total / price
            delta_qty = desired_qty - execution_pos.quantity
            trigger = "raw-action"
        else:
            delta_qty = outer_qty_after - outer_qty_before
            trigger = "outer-event"
            if abs(delta_qty) <= 1e-12:
                drift_gap = getattr(strategy, "EXECUTION_TRANSFORM_EVENT_DRIFT_GAP", None)
                if drift_gap is not None and abs(target_pct - current_pct) >= float(drift_gap):
                    desired_qty = target_pct * execution_total / price
                    delta_qty = desired_qty - execution_pos.quantity
                    trigger = "target-drift"

        if abs(delta_qty) <= 1e-12:
            continue
        notional = abs(delta_qty) * price
        if notional < max(min_notional, 1e-9):
            continue
        side = "buy" if delta_qty > 0.0 else "sell"
        qty = abs(delta_qty)
        if side == "sell":
            qty = min(qty, execution_pos.quantity)
            if qty <= 1e-12:
                continue
        estimated_after_qty = execution_pos.quantity + qty if side == "buy" else execution_pos.quantity - qty
        estimated_after_pct = estimated_after_qty * price / execution_total if execution_total > 0.0 else 0.0
        actions.append(Action(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            reason=(
                f"{strategy.name}_{side}_event-execution-transform"
                f"_{trigger}_raw{raw_pct:.0%}_target{target_pct:.0%}"
            ),
            diagnostics={
                "setup": "event-execution-transform",
                "target_pct": target_pct,
                "raw_position_pct": raw_pct,
                "transformed_target_pct": target_pct,
                "actual_position_before": current_pct,
                "actual_position_after": estimated_after_pct,
                "actual_step_pct": estimated_after_pct - current_pct,
                "execution_transform": True,
                "event_triggered_execution": True,
                "execution_trigger": trigger,
                "transform_reason": transform_reason,
                "outer_quantity_before": outer_qty_before,
                "outer_quantity_after": outer_qty_after,
                "fee_rate": fee_rate,
            },
        ))
    return actions


def financing_cost_today(strategy: PortfolioStrategyBase, portfolio: PortfolioState) -> float:
    borrowed_cash = max(0.0, -portfolio.cash)
    apr = borrow_apr(strategy)
    if apr <= 0.0 or borrowed_cash <= 0.0:
        return 0.0
    return borrowed_cash * apr / 365.25


def build_liquidation_actions(
    strategy: PortfolioStrategyBase,
    portfolio: PortfolioState,
    mark_prices: dict[str, float],
    fee_rate: float,
) -> list[Action]:
    if not force_liquidation(strategy):
        return []
    mm = maintenance_margin(strategy)
    actions: list[Action] = []
    for symbol, price in mark_prices.items():
        if price <= 0.0:
            continue
        pos = portfolio.positions.get(symbol, PositionState())
        if pos.quantity <= 1e-12:
            continue
        position_value = pos.quantity * price
        total_value = portfolio.cash + position_value
        maintenance_required = position_value * mm
        if total_value - maintenance_required > 0.0:
            continue
        actions.append(Action(
            symbol=symbol,
            side="sell",
            quantity=pos.quantity,
            price=price,
            reason=f"{strategy.name}_sell_margin-liquidation",
            diagnostics={
                "setup": "margin-liquidation",
                "target_pct": 0.0,
                "actual_position_before": position_value / total_value if total_value > 0.0 else float("inf"),
                "actual_position_after": 0.0,
                "actual_step_pct": -(position_value / total_value) if total_value > 0.0 else float("-inf"),
                "execution_transform": True,
                "margin_liquidation": True,
                "maintenance_margin": mm,
                "margin_buffer_before": total_value - maintenance_required,
                "fee_rate": fee_rate,
            },
        ))
    return actions


def audit_rows(
    strategy: PortfolioStrategyBase,
    timestamp: pd.Timestamp,
    decision_portfolio: PortfolioState,
    portfolio: PortfolioState,
    candles_by_symbol: dict[str, pd.DataFrame],
    mark_prices: dict[str, float],
    cumulative_financing: float,
    financing_cost_today_value: float,
) -> list[dict]:
    rows: list[dict] = []
    mm = maintenance_margin(strategy)
    gross_warning_level = warning_gross(strategy)
    for symbol, price in mark_prices.items():
        decision_pos = decision_portfolio.positions.get(symbol, PositionState())
        pos = portfolio.positions.get(symbol, PositionState())
        decision_total = decision_portfolio.cash + decision_pos.quantity * price
        position_value = pos.quantity * price
        total_value = portfolio.cash + position_value
        raw_pct = decision_pos.quantity * price / decision_total if decision_total > 0.0 else 0.0
        current_pct = position_value / total_value if total_value > 0.0 else 0.0
        target_pct, transform_reason = transformed_target_diagnostics_for_symbol(
            strategy=strategy,
            symbol=symbol,
            raw_position_pct=raw_pct,
            candles_by_symbol=candles_by_symbol,
            current_prices=mark_prices,
            decision_portfolio=decision_portfolio,
            execution_portfolio=portfolio,
        )
        target_pct = max(0.0, float(target_pct))
        multiplier = target_pct / raw_pct if raw_pct > 1e-12 else 1.0
        borrowed = max(0.0, -portfolio.cash)
        gross_position = position_value / total_value if total_value > 0.0 else float("inf")
        debt_to_equity = borrowed / total_value if total_value > 0.0 else float("inf")
        maintenance_required = position_value * mm
        margin_buffer = total_value - maintenance_required
        rows.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "cash": float(portfolio.cash),
            "position_value": float(position_value),
            "total_value": float(total_value),
            "raw_position_pct": float(raw_pct),
            "actual_position_pct": float(current_pct),
            "transformed_target_pct": float(target_pct),
            "transform_multiplier": float(multiplier),
            "transform_reason": str(transform_reason),
            "gross_position": float(gross_position),
            "borrowed_cash": float(borrowed),
            "debt_to_equity": float(debt_to_equity),
            "maintenance_margin": float(mm),
            "maintenance_required": float(maintenance_required),
            "margin_buffer": float(margin_buffer),
            "margin_call_est": bool(margin_buffer <= 0.0),
            "gross_warning": bool(gross_position >= gross_warning_level),
            "financing_cost_today": float(financing_cost_today_value),
            "cumulative_financing": float(cumulative_financing),
        })
    return rows
