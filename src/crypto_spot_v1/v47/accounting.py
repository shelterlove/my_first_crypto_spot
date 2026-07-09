"""Accounting boundary for V4.7 sleeve records."""

from __future__ import annotations

from dataclasses import asdict

from ..strategy_rebalance import Action
from ..v42_types import BaseSleeveState, MainSleeveState, V42Context, V42DecisionPlan, V42Sizing


class V47AccountingEngine:
    def record_sleeve_accounting(
        self,
        owner,
        context: V42Context,
        decision: V42DecisionPlan,
        sizing: V42Sizing,
        action: Action | None,
    ) -> None:
        self.ensure_sleeve_accounting(owner)
        if action is not None:
            sleeve = owner._action_sleeve(decision, sizing)
            if sleeve == "base":
                state = owner._base_sleeve_by_symbol.setdefault(context.symbol, BaseSleeveState())
            else:
                state = owner._main_sleeve_by_symbol.setdefault(context.symbol, MainSleeveState())
            before = asdict(state)
            self.apply_sleeve_fill(owner, state, action)
            if sleeve == "base":
                self.record_base_lot_fill(owner, context, decision, sizing, action)
            after = asdict(state)
            owner._sleeve_events.append({
                "timestamp": context.latest.get("timestamp"),
                "symbol": context.symbol,
                "side": action.side,
                "setup": sizing.setup,
                "sleeve": sleeve,
                "quantity": float(action.quantity),
                "price": float(action.price),
                "notional": float(action.quantity * action.price),
                "fee": float(action.quantity * action.price * owner.fee_rate),
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
        self.record_sleeve_daily_snapshot(owner, context, decision, sizing, action)

    @staticmethod
    def init_sleeve_accounting(owner) -> None:
        owner._main_sleeve_by_symbol = {}
        owner._base_sleeve_by_symbol = {}
        owner._sleeve_events = []
        owner._sleeve_daily = []
        owner._base_lots_by_symbol = {}
        owner._base_lot_events = []
        owner._next_base_lot_id_by_symbol = {}

    def ensure_sleeve_accounting(self, owner) -> None:
        if not hasattr(owner, "_sleeve_events"):
            self.init_sleeve_accounting(owner)
        if not hasattr(owner, "_base_lot_events"):
            owner._base_lots_by_symbol = {}
            owner._base_lot_events = []
            owner._next_base_lot_id_by_symbol = {}

    @staticmethod
    def apply_sleeve_fill(owner, state: MainSleeveState | BaseSleeveState, action: Action) -> None:
        qty = max(0.0, float(action.quantity))
        price = max(0.0, float(action.price))
        notional = qty * price
        fee = notional * owner.fee_rate
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
        sell_fee = sell_notional * owner.fee_rate
        cost_basis = state.avg_cost * sell_qty
        state.realized_pnl += sell_notional - sell_fee - cost_basis
        state.invested_capital = max(0.0, state.invested_capital - cost_basis)
        state.quantity = max(0.0, state.quantity - sell_qty)
        if state.quantity <= 1e-12:
            state.quantity = 0.0
            state.avg_cost = 0.0
            state.invested_capital = 0.0

    def record_base_lot_fill(
        self,
        owner,
        context: V42Context,
        decision: V42DecisionPlan,
        sizing: V42Sizing,
        action: Action,
    ) -> None:
        if action.side == "buy":
            self.record_base_lot_buy(owner, context, decision, sizing, action)
            return
        self.record_base_lot_sell(owner, context, decision, sizing, action)

    def record_base_lot_buy(
        self,
        owner,
        context: V42Context,
        decision: V42DecisionPlan,
        sizing: V42Sizing,
        action: Action,
    ) -> None:
        qty = max(0.0, float(action.quantity))
        if qty <= 0.0:
            return
        symbol = context.symbol
        next_id = int(owner._next_base_lot_id_by_symbol.get(symbol, 1))
        owner._next_base_lot_id_by_symbol[symbol] = next_id + 1
        lot_id = f"{symbol.replace('/', '')}-base-{next_id}"
        notional = qty * float(action.price)
        fee = notional * owner.fee_rate
        ledger = getattr(owner, "_base_ledger_by_symbol", {}).get(symbol, {})
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
        owner._base_lots_by_symbol.setdefault(symbol, []).append(lot)
        owner._base_lot_events.append(self.base_lot_event_row(
            owner,
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

    def record_base_lot_sell(
        self,
        owner,
        context: V42Context,
        decision: V42DecisionPlan,
        sizing: V42Sizing,
        action: Action,
    ) -> None:
        sell_qty = max(0.0, float(action.quantity))
        if sell_qty <= 0.0:
            return
        lots = owner._base_lots_by_symbol.setdefault(context.symbol, [])
        remaining_to_sell = sell_qty
        for lot in lots:
            if remaining_to_sell <= 1e-12:
                break
            available = max(0.0, float(lot.get("remaining_quantity", 0.0) or 0.0))
            if available <= 1e-12:
                continue
            consume = min(available, remaining_to_sell)
            gross = consume * float(action.price)
            fee = gross * owner.fee_rate
            cost = consume * float(lot.get("cost_per_quantity", 0.0) or 0.0)
            realized = gross - fee - cost
            lot["remaining_quantity"] = max(0.0, available - consume)
            remaining_to_sell -= consume
            owner._base_lot_events.append(self.base_lot_event_row(
                owner,
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
            owner._base_lot_events.append(self.base_lot_event_row(
                owner,
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
        owner._base_lots_by_symbol[context.symbol] = [
            lot for lot in lots if float(lot.get("remaining_quantity", 0.0) or 0.0) > 1e-12
        ]

    @staticmethod
    def base_lot_event_row(
        owner,
        *,
        context: V42Context,
        decision: V42DecisionPlan,
        sizing: V42Sizing,
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
            "fee": notional * owner.fee_rate,
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

    def record_sleeve_daily_snapshot(
        self,
        owner,
        context: V42Context,
        decision: V42DecisionPlan,
        sizing: V42Sizing,
        action: Action | None,
    ) -> None:
        main = owner._main_sleeve_by_symbol.setdefault(context.symbol, MainSleeveState())
        base = owner._base_sleeve_by_symbol.setdefault(context.symbol, BaseSleeveState())
        main_value = main.quantity * context.price
        base_value = base.quantity * context.price
        exchange_quantity_before = float(context.pos.quantity)
        exchange_quantity_after_est = self.estimate_exchange_quantity_after_action(exchange_quantity_before, action)
        sleeve_quantity = float(main.quantity + base.quantity)
        owner._sleeve_daily.append({
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
    def estimate_exchange_quantity_after_action(exchange_quantity_before: float, action: Action | None) -> float:
        if action is None:
            return exchange_quantity_before
        qty = max(0.0, float(action.quantity))
        if action.side == "buy":
            return exchange_quantity_before + qty
        return max(0.0, exchange_quantity_before - qty)

    @staticmethod
    def source_ledger(owner, symbol: str) -> dict:
        ledger = owner._base_ledger_by_symbol.setdefault(symbol, owner._new_base_ledger())
        return ledger.setdefault("source_ledger", {})

    @staticmethod
    def sync_base_from_sources(owner, context: V42Context) -> None:
        ledger = owner._base_ledger_by_symbol.setdefault(context.symbol, owner._new_base_ledger())
        sources = ledger.setdefault("source_ledger", {})
        total_qty = sum(max(0.0, float(src.get("quantity", 0.0) or 0.0)) for src in sources.values())
        ledger["base_quantity"] = total_qty
        ledger["base_position_pct"] = owner._base_pct_from_quantity(context, total_qty)
        if total_qty > 0.0:
            total_cost = sum(
                max(0.0, float(src.get("quantity", 0.0) or 0.0))
                * max(0.0, float(src.get("avg_entry", 0.0) or 0.0))
                for src in sources.values()
            )
            ledger["base_avg_entry_price"] = total_cost / total_qty if total_qty > 0.0 else 0.0
            ledger["base_source"] = "-".join(
                sorted(src for src, row in sources.items() if float(row.get("quantity", 0.0) or 0.0) > 1e-12)
            )
        else:
            ledger["base_quantity"] = 0.0
            ledger["base_position_pct"] = 0.0
            ledger["base_avg_entry_price"] = 0.0
            ledger["base_entry_call"] = None
            ledger["base_layer"] = 0
            ledger["base_peak_price"] = 0.0
            ledger["base_peak_profit"] = 0.0
            ledger["base_source"] = ""

    def add_base_source(self, owner, context: V42Context, action: Action, source: str, layer: int = 1) -> None:
        ledger = owner._base_ledger_by_symbol.setdefault(context.symbol, owner._new_base_ledger())
        sources = ledger.setdefault("source_ledger", {})
        row = sources.setdefault(source, {"quantity": 0.0, "avg_entry": 0.0, "entry_call": owner._call_count})
        old_qty = float(row.get("quantity", 0.0) or 0.0)
        old_entry = float(row.get("avg_entry", context.price) or context.price)
        add_qty = float(action.quantity)
        new_qty = old_qty + add_qty
        row["quantity"] = new_qty
        row["avg_entry"] = (old_entry * old_qty + context.price * add_qty) / new_qty if new_qty > 0.0 else 0.0
        row["entry_call"] = row.get("entry_call") or owner._call_count
        ledger["base_layer"] = max(int(ledger.get("base_layer", 0) or 0), int(layer))
        if ledger.get("base_entry_call") is None:
            ledger["base_entry_call"] = owner._call_count
            ledger["base_peak_price"] = context.price
            ledger["base_peak_profit"] = 0.0
        self.sync_base_from_sources(owner, context)

    @staticmethod
    def source_quantity(owner, context: V42Context, source: str) -> float:
        sources = owner._base_ledger_by_symbol.get(context.symbol, {}).get("source_ledger", {})
        return max(0.0, float(sources.get(source, {}).get("quantity", 0.0) or 0.0))

    def consume_base_sources(
        self,
        owner,
        context: V42Context,
        quantity: float,
        sources_allowed: tuple[str, ...],
    ) -> float:
        ledger = owner._base_ledger_by_symbol.get(context.symbol)
        if not ledger:
            return 0.0
        sources = ledger.setdefault("source_ledger", {})
        remaining = max(0.0, float(quantity))
        consumed = 0.0
        for source in sources_allowed:
            if remaining <= 1e-12:
                break
            row = sources.get(source)
            if not row:
                continue
            qty = max(0.0, float(row.get("quantity", 0.0) or 0.0))
            take = min(qty, remaining)
            row["quantity"] = max(0.0, qty - take)
            remaining -= take
            consumed += take
        self.sync_base_from_sources(owner, context)
        return consumed

    def record_protected_floor_base_buy(self, owner, context: V42Context, action: Action) -> None:
        ledger = owner._base_ledger_by_symbol.setdefault(context.symbol, owner._new_base_ledger())
        buy_pct = action.quantity * action.price / context.total_value if context.total_value > 0.0 else 0.0
        old_base_qty = float(ledger.get("base_quantity", 0.0) or 0.0)
        old_entry = float(ledger.get("base_avg_entry_price", context.price) or context.price)
        new_base_qty = old_base_qty + float(action.quantity)
        if new_base_qty > 0.0:
            ledger["base_avg_entry_price"] = (old_entry * old_base_qty + context.price * float(action.quantity)) / new_base_qty
        ledger["base_budget_used_pct"] = owner._ledger_base_budget_used_pct(ledger) + max(0.0, buy_pct)
        ledger["base_quantity"] = new_base_qty
        ledger["base_position_pct"] = owner._base_pct_from_quantity(context, new_base_qty)
        ledger["base_source"] = owner._join_guard(str(ledger.get("base_source", "")), "protected_floor")
        ledger["base_layer"] = max(int(ledger.get("base_layer", 0) or 0), 1)
        if ledger.get("base_entry_call") is None:
            ledger["base_entry_call"] = owner._call_count
            ledger["base_peak_price"] = context.price
            ledger["base_peak_profit"] = 0.0

        old_qty = float(ledger.get("protected_floor_quantity", 0.0) or 0.0)
        old_entry = float(ledger.get("protected_floor_avg_entry_price", context.price) or context.price)
        add_qty = float(action.quantity)
        new_qty = old_qty + add_qty
        if new_qty > 0.0:
            ledger["protected_floor_avg_entry_price"] = (old_entry * old_qty + context.price * add_qty) / new_qty
        ledger["protected_floor_quantity"] = new_qty
        ledger["protected_floor_entry_call"] = ledger.get("protected_floor_entry_call") or owner._call_count
        ledger["protected_floor_last_exit_call"] = ledger.get("protected_floor_last_exit_call")
        ledger["protected_floor_last_exit_price"] = float(ledger.get("protected_floor_last_exit_price", 0.0) or 0.0)

        self.add_base_source(owner, context, action, "protected_floor", layer=1)
        source_qty = self.source_quantity(owner, context, "protected_floor")
        source = ledger.get("source_ledger", {}).get("protected_floor", {})
        ledger["protected_floor_quantity"] = source_qty
        ledger["protected_floor_avg_entry_price"] = float(source.get("avg_entry", 0.0) or 0.0)
        ledger["protected_floor_entry_call"] = source.get("entry_call") or owner._call_count

    def record_bear_base_buy(self, owner, context: V42Context, action: Action) -> None:
        recovery_ledger = owner._recovery_ledger_by_symbol.get(context.symbol)
        if recovery_ledger is None:
            return
        ledger = owner._base_ledger_by_symbol.setdefault(context.symbol, owner._new_base_ledger())
        buy_pct = action.quantity * action.price / context.total_value if context.total_value > 0.0 else 0.0
        buy_pct = max(0.0, buy_pct)
        old_base_qty = float(ledger.get("base_quantity", 0.0) or 0.0)
        old_entry = float(ledger.get("base_avg_entry_price", context.price) or context.price)
        new_base_qty = old_base_qty + float(action.quantity)
        if new_base_qty > 0.0:
            ledger["base_avg_entry_price"] = (old_entry * old_base_qty + context.price * float(action.quantity)) / new_base_qty
        ledger["base_budget_used_pct"] = owner._ledger_base_budget_used_pct(ledger) + buy_pct
        ledger["base_quantity"] = new_base_qty
        ledger["base_position_pct"] = owner._base_pct_from_quantity(context, new_base_qty)
        if ledger.get("base_entry_call") is None:
            ledger["base_entry_call"] = owner._call_count
            ledger["base_peak_price"] = context.price
            ledger["base_peak_profit"] = 0.0
        proposal = owner._bear_base_buy_proposal(
            context,
            owner._recovery_ledger_by_symbol[context.symbol],
            ledger,
            owner._bear_base_location_depth(context),
        )
        layer = int(proposal.layer)
        ledger["base_layer"] = max(int(ledger.get("base_layer", 0) or 0), layer)
        if layer == 1:
            owner._diag["v4_2_bear_base_layer1_buy_count"] += 1
        elif layer == 2:
            owner._diag["v4_2_bear_base_layer2_buy_count"] += 1

        self.add_base_source(owner, context, action, "strategic_base", layer=layer)

    def record_base_led_recovery_buy(self, owner, context: V42Context, action: Action) -> None:
        self.add_base_source(owner, context, action, "base_led_recovery", layer=2)

    def record_protected_floor_base_exit(self, owner, context: V42Context, action: Action) -> None:
        ledger = owner._base_ledger_by_symbol.get(context.symbol)
        if not ledger:
            return
        consumed = self.consume_base_sources(owner, context, float(action.quantity), ("protected_floor",))
        ledger["protected_floor_quantity"] = self.source_quantity(owner, context, "protected_floor")
        ledger["protected_floor_last_exit_call"] = owner._call_count
        ledger["protected_floor_last_exit_price"] = context.price
        if consumed <= 1e-12:
            return

    def record_base_exit_sell(self, owner, context: V42Context, action: Action) -> None:
        ledger = owner._base_ledger_by_symbol.get(context.symbol)
        if ledger and "source_ledger" in ledger:
            allowed = ("strategic_base",)
            if context.symbol != "BNB/USDT":
                allowed = ("strategic_base", "protected_floor", "base_led_recovery")
            consumed = self.consume_base_sources(owner, context, float(action.quantity), allowed)
            if consumed > 1e-12:
                owner._diag["v4_2_bear_base_exit_count"] += 1
            return

        if ledger is None:
            return
        base_qty = owner._base_quantity(context)
        if base_qty <= 0.0 or action.quantity <= 0.0:
            return
        layer = int(ledger.get("base_layer", 0) or 0)
        ledger["base_quantity"] = max(0.0, base_qty - float(action.quantity))
        ledger["base_position_pct"] = owner._base_pct_from_quantity(context, float(ledger.get("base_quantity", 0.0) or 0.0))
        if float(ledger.get("base_quantity", 0.0) or 0.0) <= 1e-12:
            ledger["base_quantity"] = 0.0
            ledger["base_position_pct"] = 0.0
            ledger["base_avg_entry_price"] = 0.0
            ledger["base_entry_call"] = None
            ledger["base_layer"] = 0
            ledger["base_peak_price"] = 0.0
            ledger["base_peak_profit"] = 0.0
        owner._diag["v4_2_bear_base_exit_count"] += 1
        if layer == 1:
            owner._diag["v4_2_bear_base_layer1_exit_count"] += 1
        elif layer == 2:
            owner._diag["v4_2_bear_base_layer2_exit_count"] += 1
