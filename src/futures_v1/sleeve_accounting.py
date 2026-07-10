from __future__ import annotations

from dataclasses import asdict

from .strategy_rebalance import Action
from .strategy_types import BaseSleeveState, MainSleeveState, StrategyContext, StrategyDecisionPlan, StrategySizing


class StrategySleeveAccountingMixin:
    def _init_sleeve_accounting(self) -> None:
        self._main_sleeve_by_symbol: dict[str, MainSleeveState] = {}
        self._base_sleeve_by_symbol: dict[str, BaseSleeveState] = {}
        self._sleeve_events: list[dict] = []
        self._sleeve_daily: list[dict] = []
        self._base_lots_by_symbol: dict[str, list[dict]] = {}
        self._base_lot_events: list[dict] = []
        self._next_base_lot_id_by_symbol: dict[str, int] = {}

    def strategy_sleeve_events(self) -> list[dict]:
        return list(getattr(self, "_sleeve_events", []))

    def strategy_sleeve_daily(self) -> list[dict]:
        return list(getattr(self, "_sleeve_daily", []))

    def strategy_base_lot_events(self) -> list[dict]:
        return list(getattr(self, "_base_lot_events", []))

    def _record_sleeve_accounting(
        self,
        context: StrategyContext,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        self._ensure_sleeve_accounting()
        if action is not None:
            sleeve = self._action_sleeve(decision, sizing)
            if sleeve == "base":
                state = self._base_sleeve_by_symbol.setdefault(context.symbol, BaseSleeveState())
            else:
                state = self._main_sleeve_by_symbol.setdefault(context.symbol, MainSleeveState())
            before = asdict(state)
            self._apply_sleeve_fill(state, action)
            if sleeve == "base":
                self._record_base_lot_fill(context, decision, sizing, action)
            after = asdict(state)
            self._sleeve_events.append({
                "timestamp": context.latest.get("timestamp"),
                "symbol": context.symbol,
                "side": action.side,
                "setup": sizing.setup,
                "sleeve": sleeve,
                "quantity": float(action.quantity),
                "price": float(action.price),
                "notional": float(action.quantity * action.price),
                "fee": float(action.quantity * action.price * self.fee_rate),
                "main_intent": str(getattr(decision, "main_intent", decision.intent)),
                "base_intent": str(getattr(decision, "base_intent", "")),
                "main_delta": float(getattr(decision, "main_delta", 0.0)),
                "base_delta": float(getattr(decision, "base_delta", 0.0)),
                "quantity_before": float(before["quantity"]),
                "avg_cost_before": float(before["avg_cost"]),
                "realized_pnl_before": float(before["realized_pnl"]),
                "invested_capital_before": float(before["invested_capital"]),
                "quantity_after": float(after["quantity"]),
                "avg_cost_after": float(after["avg_cost"]),
                "realized_pnl_after": float(after["realized_pnl"]),
                "invested_capital_after": float(after["invested_capital"]),
            })
        self._record_sleeve_daily_snapshot(context, decision, sizing, action)

    def _ensure_sleeve_accounting(self) -> None:
        if not hasattr(self, "_sleeve_events"):
            self._init_sleeve_accounting()
        if not hasattr(self, "_base_lot_events"):
            self._base_lots_by_symbol = {}
            self._base_lot_events = []
            self._next_base_lot_id_by_symbol = {}

    def _action_sleeve(self, decision: StrategyDecisionPlan, sizing: StrategySizing) -> str:
        primary = decision.primary_sleeve.sleeve if decision.primary_sleeve is not None else ""
        if primary in {"bear-base", "bear-base-exit"}:
            return "base"
        if sizing.setup == "bear-base-exit" or "core_bear_base" in str(sizing.guard):
            return "base"
        return "main"

    def _apply_sleeve_fill(self, state: MainSleeveState | BaseSleeveState, action: Action) -> None:
        qty = max(0.0, float(action.quantity))
        price = max(0.0, float(action.price))
        notional = qty * price
        fee = notional * self.fee_rate
        if qty <= 0.0:
            return
        if action.side == "buy":
            total_cost = state.avg_cost * state.quantity + notional + fee
            state.quantity += qty
            state.avg_cost = total_cost / state.quantity if state.quantity > 0.0 else 0.0
            state.invested_capital += notional + fee
            return

        sell_qty = min(qty, state.quantity)
        if sell_qty <= 0.0:
            return
        sell_notional = sell_qty * price
        sell_fee = sell_notional * self.fee_rate
        cost_basis = state.avg_cost * sell_qty
        state.realized_pnl += sell_notional - sell_fee - cost_basis
        state.invested_capital = max(0.0, state.invested_capital - cost_basis)
        state.quantity = max(0.0, state.quantity - sell_qty)
        if state.quantity <= 1e-12:
            state.quantity = 0.0
            state.avg_cost = 0.0
            state.invested_capital = 0.0

    def _record_base_lot_fill(
        self,
        context: StrategyContext,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action,
    ) -> None:
        if action.side == "buy":
            self._record_base_lot_buy(context, decision, sizing, action)
            return
        self._record_base_lot_sell(context, decision, sizing, action)

    def _record_base_lot_buy(
        self,
        context: StrategyContext,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action,
    ) -> None:
        qty = max(0.0, float(action.quantity))
        if qty <= 0.0:
            return
        symbol = context.symbol
        next_id = int(self._next_base_lot_id_by_symbol.get(symbol, 1))
        self._next_base_lot_id_by_symbol[symbol] = next_id + 1
        lot_id = f"{symbol.replace('/', '')}-base-{next_id}"
        notional = qty * float(action.price)
        fee = notional * self.fee_rate
        ledger = getattr(self, "_base_ledger_by_symbol", {}).get(symbol, {})
        lot = {
            "lot_id": lot_id,
            "symbol": symbol,
            "entry_timestamp": context.latest.get("timestamp"),
            "entry_setup": sizing.setup,
            "entry_price": float(action.price),
            "entry_quantity": qty,
            "remaining_quantity": qty,
            "entry_notional": notional,
            "entry_fee": fee,
            "entry_cost": notional + fee,
            "cost_per_quantity": (notional + fee) / qty if qty > 0.0 else 0.0,
            "layer": int(ledger.get("base_layer", 0) or 0),
        }
        self._base_lots_by_symbol.setdefault(symbol, []).append(lot)
        self._base_lot_events.append(self._base_lot_event_row(
            context=context,
            decision=decision,
            sizing=sizing,
            action=action,
            lot=lot,
            event="buy",
            quantity=qty,
            realized_pnl=0.0,
            remaining_quantity=qty,
        ))

    def _record_base_lot_sell(
        self,
        context: StrategyContext,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action,
    ) -> None:
        sell_qty = max(0.0, float(action.quantity))
        if sell_qty <= 0.0:
            return
        lots = self._base_lots_by_symbol.setdefault(context.symbol, [])
        remaining_to_sell = sell_qty
        for lot in lots:
            if remaining_to_sell <= 1e-12:
                break
            available = max(0.0, float(lot.get("remaining_quantity", 0.0) or 0.0))
            if available <= 1e-12:
                continue
            consume = min(available, remaining_to_sell)
            gross = consume * float(action.price)
            fee = gross * self.fee_rate
            cost = consume * float(lot.get("cost_per_quantity", 0.0) or 0.0)
            realized = gross - fee - cost
            lot["remaining_quantity"] = max(0.0, available - consume)
            remaining_to_sell -= consume
            self._base_lot_events.append(self._base_lot_event_row(
                context=context,
                decision=decision,
                sizing=sizing,
                action=action,
                lot=lot,
                event="sell",
                quantity=consume,
                realized_pnl=realized,
                remaining_quantity=float(lot["remaining_quantity"]),
            ))
        if remaining_to_sell > 1e-10:
            self._base_lot_events.append(self._base_lot_event_row(
                context=context,
                decision=decision,
                sizing=sizing,
                action=action,
                lot={},
                event="sell_unmatched",
                quantity=remaining_to_sell,
                realized_pnl=0.0,
                remaining_quantity=0.0,
            ))
        self._base_lots_by_symbol[context.symbol] = [
            lot for lot in lots if float(lot.get("remaining_quantity", 0.0) or 0.0) > 1e-12
        ]

    def _base_lot_event_row(
        self,
        *,
        context: StrategyContext,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action,
        lot: dict,
        event: str,
        quantity: float,
        realized_pnl: float,
        remaining_quantity: float,
    ) -> dict:
        price = float(action.price)
        notional = float(quantity) * price
        return {
            "timestamp": context.latest.get("timestamp"),
            "symbol": context.symbol,
            "event": event,
            "side": action.side,
            "setup": sizing.setup,
            "lot_id": str(lot.get("lot_id", "")),
            "layer": int(lot.get("layer", 0) or 0),
            "quantity": float(quantity),
            "price": price,
            "notional": notional,
            "fee": notional * self.fee_rate,
            "realized_pnl": float(realized_pnl),
            "remaining_quantity": float(remaining_quantity),
            "entry_timestamp": lot.get("entry_timestamp"),
            "entry_setup": str(lot.get("entry_setup", "")),
            "entry_price": float(lot.get("entry_price", 0.0) or 0.0),
            "entry_quantity": float(lot.get("entry_quantity", 0.0) or 0.0),
            "entry_notional": float(lot.get("entry_notional", 0.0) or 0.0),
            "entry_fee": float(lot.get("entry_fee", 0.0) or 0.0),
            "entry_cost": float(lot.get("entry_cost", 0.0) or 0.0),
            "cost_per_quantity": float(lot.get("cost_per_quantity", 0.0) or 0.0),
            "main_intent": str(getattr(decision, "main_intent", decision.intent)),
            "base_intent": str(getattr(decision, "base_intent", "")),
            "main_delta": float(getattr(decision, "main_delta", 0.0)),
            "base_delta": float(getattr(decision, "base_delta", 0.0)),
        }

    def _record_sleeve_daily_snapshot(
        self,
        context: StrategyContext,
        decision: StrategyDecisionPlan,
        sizing: StrategySizing,
        action: Action | None,
    ) -> None:
        main = self._main_sleeve_by_symbol.setdefault(context.symbol, MainSleeveState())
        base = self._base_sleeve_by_symbol.setdefault(context.symbol, BaseSleeveState())
        main_value = main.quantity * context.price
        base_value = base.quantity * context.price
        exchange_quantity_before = float(context.pos.quantity)
        exchange_quantity_after_est = self._estimate_exchange_quantity_after_action(
            exchange_quantity_before,
            action,
        )
        sleeve_quantity = float(main.quantity + base.quantity)
        self._sleeve_daily.append({
            "timestamp": context.latest.get("timestamp"),
            "symbol": context.symbol,
            "price": float(context.price),
            "main_quantity": float(main.quantity),
            "main_avg_cost": float(main.avg_cost),
            "main_value": float(main_value),
            "main_invested_capital": float(main.invested_capital),
            "main_realized_pnl": float(main.realized_pnl),
            "main_unrealized_pnl": float(main_value - main.avg_cost * main.quantity),
            "base_quantity": float(base.quantity),
            "base_avg_cost": float(base.avg_cost),
            "base_value": float(base_value),
            "base_invested_capital": float(base.invested_capital),
            "base_realized_pnl": float(base.realized_pnl),
            "base_unrealized_pnl": float(base_value - base.avg_cost * base.quantity),
            "exchange_quantity": exchange_quantity_before,
            "exchange_quantity_before": exchange_quantity_before,
            "exchange_quantity_after_est": exchange_quantity_after_est,
            "sleeve_quantity": sleeve_quantity,
            "quantity_diff_before": exchange_quantity_before - sleeve_quantity,
            "quantity_diff_after_est": exchange_quantity_after_est - sleeve_quantity,
            "snapshot_phase": "post_fill_estimated" if action is not None else "current",
            "main_intent": str(getattr(decision, "main_intent", decision.intent)),
            "base_intent": str(getattr(decision, "base_intent", "")),
            "main_delta": float(getattr(decision, "main_delta", 0.0)),
            "base_delta": float(getattr(decision, "base_delta", 0.0)),
            "action_side": "" if action is None else action.side,
            "action_setup": sizing.setup,
            "blocked_reason": sizing.blocked_reason,
        })

    @staticmethod
    def _estimate_exchange_quantity_after_action(
        exchange_quantity_before: float,
        action: Action | None,
    ) -> float:
        if action is None:
            return exchange_quantity_before
        qty = max(0.0, float(action.quantity))
        if action.side == "buy":
            return exchange_quantity_before + qty
        return max(0.0, exchange_quantity_before - qty)
