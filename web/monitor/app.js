const DATA_URL = "data/monitor.json";

const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 });
const pct = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });

document.getElementById("refreshButton").addEventListener("click", () => loadData(true));
loadData(false);

async function loadData(force) {
  const url = force ? `${DATA_URL}?t=${Date.now()}` : DATA_URL;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    render(data);
  } catch (error) {
    renderError(error);
  }
}

function render(data) {
  document.getElementById("subtitle").textContent =
    `${data.strategy || "strategy"} | ${joinSymbols(data.symbols)} | generated ${shortTime(data.generated_at)}`;
  renderStatus(data);
  renderAlerts(data.alerts || []);
  renderSignals(data.latest_signal);
  renderPositions(data.latest_futures_report);
  renderOrders(data.latest_futures_report, data.latest_spot_report);
  renderReports(data);
}

function renderStatus(data) {
  const futures = data.latest_futures_report || {};
  const spot = data.latest_spot_report || {};
  const signal = data.latest_signal || {};
  const risk = futures.risk || {};
  const items = [
    ["Futures Mode", futures.mode || "none", futures.path || "No futures report"],
    ["Wallet USDT", money(futures.wallet_balance_usdt || spot.total_equity_usdt), "Latest reported equity"],
    ["Orders", String((futures.order_count || 0) + (spot.order_count || 0)), "Latest planned orders"],
    ["Liq. Buffer", risk.min_liquidation_buffer_pct == null ? "n/a" : pct.format(Number(risk.min_liquidation_buffer_pct)), "Minimum futures buffer"],
    ["Target Gross Cap", futures.target_gross_cap || "n/a", `Exchange leverage ${futures.exchange_leverage || "n/a"}`],
    ["Signal Age", age(signal.mtime), signal.path || "No signal file"],
    ["Futures Age", age(futures.mtime), futures.path || "No futures report"],
    ["Spot Age", age(spot.mtime), spot.path || "No spot report"],
  ];
  document.getElementById("statusGrid").innerHTML = items.map(([label, value, hint]) => `
    <div class="metric">
      <div class="label">${escapeHtml(label)}</div>
      <div class="value">${escapeHtml(value)}</div>
      <div class="hint" title="${escapeHtml(hint)}">${escapeHtml(hint)}</div>
    </div>
  `).join("");
}

function renderAlerts(alerts) {
  const node = document.getElementById("alerts");
  if (!alerts.length) {
    node.innerHTML = `<div class="alert info">No dashboard alerts from latest files.</div>`;
    return;
  }
  node.innerHTML = alerts.map(alert => `
    <div class="alert ${escapeHtml(alert.level || "info")}">${escapeHtml(alert.message || "")}</div>
  `).join("");
}

function renderSignals(signal) {
  document.getElementById("signalPath").textContent = signal ? signal.path : "No signal";
  const rows = signal && Array.isArray(signal.rows) ? signal.rows : [];
  document.getElementById("signalsBody").innerHTML = rows.length ? rows.map(row => `
    <tr>
      <td>${escapeHtml(row.symbol)}</td>
      <td><span class="badge ${escapeHtml(String(row.action || "").toLowerCase())}">${escapeHtml(row.action || "hold")}</span></td>
      <td>${formatMaybePct(row.current_pct)}</td>
      <td>${formatMaybePct(row.target_pct)}</td>
      <td>${escapeHtml(row.confirmed_state || row.raw_state || "")}</td>
      <td class="reason" title="${escapeHtml(row.no_trade_reason || row.reason || "")}">${escapeHtml(row.no_trade_reason || row.reason || "")}</td>
    </tr>
  `).join("") : emptyRow(6, "No signal rows.");
}

function renderPositions(report) {
  document.getElementById("futuresPath").textContent = report ? report.path : "No futures report";
  const positions = report && report.positions ? Object.values(report.positions).filter(Boolean) : [];
  document.getElementById("positionsBody").innerHTML = positions.length ? positions.map(position => `
    <tr>
      <td>${escapeHtml(position.symbol)}</td>
      <td>${num(position.position_amt)}</td>
      <td>${money(position.mark_price)}</td>
      <td>${money(position.liquidation_price)}</td>
      <td>${position.liquidation_buffer_pct == null ? "n/a" : pct.format(Number(position.liquidation_buffer_pct))}</td>
      <td>${escapeHtml(position.leverage || "")}</td>
    </tr>
  `).join("") : emptyRow(6, "No futures positions.");
}

function renderOrders(futures, spot) {
  const rows = [];
  if (futures && Array.isArray(futures.orders)) {
    futures.orders.forEach(order => rows.push(["Futures", futures.mode, order]));
  }
  if (spot && Array.isArray(spot.orders)) {
    spot.orders.forEach(order => rows.push(["Spot", spot.mode, order]));
  }
  document.getElementById("ordersBody").innerHTML = rows.length ? rows.map(([venue, mode, order]) => `
    <tr>
      <td>${venue}</td>
      <td>${escapeHtml(order.symbol || order.api_symbol || "")}</td>
      <td><span class="badge ${escapeHtml(String(order.side || "").toLowerCase())}">${escapeHtml(order.side || "")}</span></td>
      <td>${escapeHtml(order.quantity || order.quote_order_qty || "")}</td>
      <td>${money(order.notional)}</td>
      <td>${escapeHtml(order.target_gross || "")}</td>
      <td>${escapeHtml(order.clip_reason || "")}</td>
      <td><span class="badge ${escapeHtml(String(mode || "").toLowerCase())}">${escapeHtml(mode || "")}</span></td>
    </tr>
  `).join("") : emptyRow(8, "No planned orders.");
}

function renderReports(data) {
  const reports = []
    .concat((data.recent_futures_reports || []).map(report => ["Futures", report]))
    .concat((data.recent_spot_reports || []).map(report => ["Spot", report]))
    .sort((a, b) => new Date(b[1].mtime || 0) - new Date(a[1].mtime || 0))
    .slice(0, 12);
  document.getElementById("reportList").innerHTML = reports.length ? reports.map(([venue, report]) => `
    <div class="report-item">
      <div class="report-title">
        <span>${venue}</span>
        <span class="badge ${escapeHtml(String(report.mode || "").toLowerCase())}">${escapeHtml(report.mode || "")}</span>
      </div>
      <div class="report-meta">${escapeHtml(shortTime(report.mtime))}</div>
      <div class="report-meta">${escapeHtml(report.path || "")}</div>
      <div class="report-meta">orders ${report.order_count || 0} | responses ${report.response_count || 0}</div>
    </div>
  `).join("") : `<div class="report-item">No reports found.</div>`;
}

function renderError(error) {
  document.getElementById("subtitle").textContent = "Dashboard data could not be loaded.";
  document.getElementById("alerts").innerHTML =
    `<div class="alert danger">Failed to load ${DATA_URL}: ${escapeHtml(error.message)}. Run scripts/build_monitor_dashboard_data.py and serve web/monitor over HTTP.</div>`;
}

function emptyRow(cols, text) {
  return `<tr><td colspan="${cols}" class="muted">${escapeHtml(text)}</td></tr>`;
}

function formatMaybePct(value) {
  if (value === null || value === undefined || value === "") return "";
  const n = Number(value);
  return Number.isFinite(n) ? pct.format(n) : escapeHtml(String(value));
}

function money(value) {
  if (value === null || value === undefined || value === "") return "n/a";
  const n = Number(value);
  return Number.isFinite(n) ? fmt.format(n) : String(value);
}

function num(value) {
  if (value === null || value === undefined || value === "") return "0";
  const n = Number(value);
  return Number.isFinite(n) ? fmt.format(n) : String(value);
}

function age(timestamp) {
  if (!timestamp) return "n/a";
  const diff = Math.max(0, Date.now() - new Date(timestamp).getTime());
  const hours = diff / 36e5;
  if (hours < 1) return `${Math.round(hours * 60)}m`;
  if (hours < 48) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}

function shortTime(timestamp) {
  if (!timestamp) return "n/a";
  return new Date(timestamp).toLocaleString();
}

function joinSymbols(symbols) {
  return Array.isArray(symbols) ? symbols.join(", ") : "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
