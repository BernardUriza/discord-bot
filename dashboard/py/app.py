"""Insult Dashboard — entry point. Fetches metrics and renders UI."""

from browser import document, timer, ajax

from .config import METRICS_URL, LOGS_URL, TRACES_URL, FACTS_URL, REFRESH_INTERVAL
from .carousel import CardCarousel

document.select_one(".logo span").text = "INSULT"

# ── State ────────────────────────────────────────────────────────

_metrics = {}
_logs = []
_traces = []
_facts = []
_current_filter = "all"
_current_tab = "monitor"


# ── Data Fetching ────────────────────────────────────────────────

def fetch_data():
    """Fetch metrics and logs from Azure Blob."""
    req = ajax.Ajax()
    req.open("GET", f"{METRICS_URL}?t={__import__('time').time()}", True)
    req.bind("complete", _on_metrics)
    req.send()

    req2 = ajax.Ajax()
    req2.open("GET", f"{LOGS_URL}?t={__import__('time').time()}", True)
    req2.bind("complete", _on_logs)
    req2.send()

    req3 = ajax.Ajax()
    req3.open("GET", f"{TRACES_URL}?t={__import__('time').time()}", True)
    req3.bind("complete", _on_traces)
    req3.send()

    req4 = ajax.Ajax()
    req4.open("GET", f"{FACTS_URL}?t={__import__('time').time()}", True)
    req4.bind("complete", _on_facts)
    req4.send()


def _on_metrics(req):
    global _metrics
    if req.status == 200:
        import json
        _metrics = json.loads(req.text)
        _render_metrics()
        _update_status("live", "connected")
    else:
        _update_status("error", f"HTTP {req.status}")


def _on_logs(req):
    global _logs
    if req.status == 200:
        import json
        _logs = json.loads(req.text)
        _render_logs()


def _on_traces(req):
    global _traces
    if req.status == 200:
        import json
        _traces = json.loads(req.text)
        _render_traces()


# ── Trace Carousel ───────────────────────────────────────────────

def _render_trace_card(trace):
    """Render a single message trace as a carousel card."""
    from browser import html

    card = html.DIV()

    # User + timestamp
    import time
    ts = trace.get("ts", 0)
    time_str = time.strftime("%H:%M", time.localtime(ts)) if ts else "??:??"
    card <= html.DIV(f"{trace.get('user', '?')} · {time_str}", Class="trace-user")

    # Input
    inp = trace.get("input", "")
    card <= html.DIV(f'"{inp}"' if inp else "(empty)", Class="trace-input")

    # Response
    resp = trace.get("response", "")
    card <= html.DIV(resp if resp else "(no text)", Class="trace-response")

    # Tags
    meta = html.DIV(Class="trace-meta")
    meta <= html.SPAN(trace.get("preset", "?"), Class="trace-tag preset")
    pressure = trace.get("pressure", 0)
    if pressure:
        meta <= html.SPAN(f"P{pressure}", Class="trace-tag pressure")
    shape = trace.get("expression_shape", "")
    if shape:
        meta <= html.SPAN(shape, Class="trace-tag shape")
    for tool in trace.get("tools", []):
        meta <= html.SPAN(tool, Class="trace-tag tool")
    if trace.get("character_break"):
        meta <= html.SPAN("BREAK!", Class="trace-tag break")
    if trace.get("anti_pattern"):
        meta <= html.SPAN("DRIFT", Class="trace-tag break")
    if trace.get("reactions"):
        meta <= html.SPAN(" ".join(trace["reactions"][:3]), Class="trace-tag")
    card <= meta

    return card


_trace_carousel = CardCarousel(
    container_id="traces-carousel",
    render_card=_render_trace_card,
    get_id=lambda t: str(t.get("ts", 0)),
    empty_msg="No messages yet.",
)


def _render_traces():
    # Most recent first
    _trace_carousel.render(list(reversed(_traces[-30:])))


# ── Rendering ────────────────────────────────────────────────────

def _render_metrics():
    if not _metrics:
        return

    counters = _metrics.get("counters", {})
    bot = _metrics.get("bot", {})
    db = _metrics.get("db", {})

    # Stat cards
    uptime_s = _metrics.get("uptime_seconds", 0)
    hours = uptime_s // 3600
    mins = (uptime_s % 3600) // 60
    document["val-uptime"].text = f"{hours}h {mins}m"
    document["val-messages"].text = str(db.get("total_messages", counters.get("messages_total", 0)))
    document["val-latency"].text = f"{bot.get('latency_ms', 0)}ms"
    document["val-errors"].text = str(counters.get("llm_errors", 0))

    # Preset distribution bars
    _render_presets(counters)

    # Counters list
    _render_counters(counters)


def _render_presets(counters):
    container = document["preset-bars"]
    container.clear()

    presets = [
        ("default_abrasive", "#e74c3c", "Abrasive"),
        ("playful_roast", "#f39c12", "Roast"),
        ("intellectual_pressure", "#3498db", "Intellectual"),
        ("relational_probe", "#9b59b6", "Relational"),
        ("respectful_serious", "#2ecc71", "Serious"),
        ("meta_deflection", "#95a5a6", "Meta"),
    ]

    total = sum(counters.get(f"preset_{p[0]}", 0) for p in presets) or 1

    from browser import html
    for key, color, label in presets:
        count = counters.get(f"preset_{key}", 0)
        pct = round(count / total * 100)

        row = html.DIV(Class="preset-row")
        row <= html.SPAN(f"{label}", Class="preset-label")
        bar_bg = html.DIV(Class="preset-bar-bg")
        bar = html.DIV(Class="preset-bar-fill", style={"width": f"{pct}%", "background": color})
        bar_bg <= bar
        row <= bar_bg
        row <= html.SPAN(f"{count} ({pct}%)", Class="preset-count")
        container <= row


def _render_counters(counters):
    container = document["counter-list"]
    container.clear()

    from browser import html
    items = [
        ("LLM Requests", counters.get("llm_requests", 0)),
        ("Character Breaks", counters.get("character_breaks", 0)),
        ("Anti-patterns", counters.get("anti_patterns", 0)),
        ("Whisper Transcriptions", counters.get("whisper_transcriptions", 0)),
        ("Reminders Created", counters.get("reminders_created", 0)),
        ("Facts Extracted", counters.get("facts_extracted", 0)),
        ("Facts Failed", counters.get("facts_failed", 0)),
    ]

    for label, value in items:
        row = html.DIV(Class="counter-row")
        row <= html.SPAN(label, Class="counter-label")
        row <= html.SPAN(str(value), Class="counter-value")
        container <= row


def _render_logs():
    container = document["log-entries"]
    container.clear()

    from browser import html
    import time

    filtered = _logs if _current_filter == "all" else [
        e for e in _logs if _current_filter in e.get("event", "")
    ]

    # Show most recent first, max 100
    for entry in reversed(filtered[-100:]):
        ts = entry.get("ts", 0)
        time_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "??:??:??"
        event = entry.get("event", "unknown")
        level = entry.get("level", "info")

        row = html.DIV(Class=f"log-row log-{level}")
        row <= html.SPAN(time_str, Class="log-time")
        row <= html.SPAN(event, Class="log-event")

        # Extra info
        extras = {k: v for k, v in entry.items() if k not in ("ts", "event", "level", "timestamp")}
        if extras:
            extra_str = " ".join(f"{k}={v}" for k, v in list(extras.items())[:4])
            row <= html.SPAN(extra_str, Class="log-extra")

        container <= row

    # Auto-scroll to bottom
    container.scrollTop = container.scrollHeight


def _update_status(state, text):
    el = document["status-indicator"]
    el.text = text
    el.className = f"status status-{state}"


# ── Filter binding ───────────────────────────────────────────────

def _on_filter_change(ev):
    global _current_filter
    _current_filter = document["log-filter"].value
    _render_logs()


document["log-filter"].bind("change", _on_filter_change)


# ── Facts View ───────────────────────────────────────────────────

def _on_facts(req):
    global _facts
    if req.status == 200:
        import json
        _facts = json.loads(req.text)
        if _current_tab == "facts":
            _render_facts()


def _render_facts():
    from browser import html
    import time

    grid = document["facts-grid"]
    grid.clear()

    if not _facts:
        grid <= html.P("No facts yet.", Class="empty-msg")
        return

    # Get filter values
    user_filter = document["facts-user-filter"].value
    cat_filter = document["facts-category-filter"].value

    # Group by user
    users = {}
    categories = set()
    for f in _facts:
        uid = f.get("user_id", "?")
        if user_filter != "all" and uid != user_filter:
            continue
        cat = f.get("category", "general")
        categories.add(cat)
        if cat_filter != "all" and cat != cat_filter:
            continue
        users.setdefault(uid, []).append(f)

    # Update stats
    document["facts-stats"].text = f"{len(_facts)} facts · {len(set(f.get('user_id','') for f in _facts))} users"

    # Update filter dropdowns (preserve selection)
    _update_dropdown("facts-user-filter", sorted(set(f.get("user_id", "?") for f in _facts)), user_filter)
    _update_dropdown("facts-category-filter", sorted(categories), cat_filter)

    # Render cards per user
    for uid in sorted(users.keys()):
        facts = users[uid]
        card = html.DIV(Class="fact-user-card")

        header = html.DIV(Class="fact-user-header")
        header <= html.SPAN(uid, Class="fact-user-name")
        header <= html.SPAN(f"{len(facts)} facts", Class="fact-user-count")
        card <= header

        for f in facts:
            item = html.DIV(Class="fact-item")
            cat = f.get("category", "general")
            item <= html.SPAN(cat, Class=f"fact-category {cat}")
            item <= html.SPAN(f.get("fact", ""), Class="fact-text")
            ts = f.get("updated_at", 0)
            if ts:
                time_str = time.strftime("%m/%d", time.localtime(float(ts)))
                item <= html.SPAN(time_str, Class="fact-time")
            card <= item

        grid <= card


def _update_dropdown(element_id, values, current):
    """Update a select dropdown with values, preserving current selection."""
    from browser import html
    sel = document[element_id]
    sel.clear()
    sel <= html.OPTION("All", value="all")
    for v in values:
        opt = html.OPTION(v, value=v)
        if v == current:
            opt.attrs["selected"] = "selected"
        sel <= opt


# ── Tab switching ────────────────────────────────────────────────

def _switch_tab(ev):
    global _current_tab
    tab = ev.target.attrs.get("data-tab", "monitor")
    _current_tab = tab

    # Update tab buttons
    for t in document.select(".tab"):
        t.classList.remove("active")
    ev.target.classList.add("active")

    # Toggle views
    if tab == "monitor":
        document.select_one(".bento").style.display = "grid"
        document["facts-view"].style.display = "none"
    elif tab == "facts":
        document.select_one(".bento").style.display = "none"
        document["facts-view"].style.display = ""
        _render_facts()


document["tab-monitor"].bind("click", _switch_tab)
document["tab-facts"].bind("click", _switch_tab)
document["facts-user-filter"].bind("change", lambda ev: _render_facts())
document["facts-category-filter"].bind("change", lambda ev: _render_facts())


# ── Auto-refresh ─────────────────────────────────────────────────

fetch_data()
timer.set_interval(fetch_data, REFRESH_INTERVAL)
