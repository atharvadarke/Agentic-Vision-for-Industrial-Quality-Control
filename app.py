"""
Gradio Dashboard — Agentic Vision for Industrial Quality Control
=================================================================
A production-quality web UI that:
  • Sends part images to the FastAPI backend (POST /inspect-part)
  • Displays structured inspection results (Pass/Fail, agent report)
  • Reads defect history directly from the local SQLite DB
  • Shows live system-health status from GET /health
  • Exposes raw Prometheus metrics from GET /metrics

Run this alongside the FastAPI server:
    Terminal 1: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
    Terminal 2: python app.py
"""

import os
import sys
import sqlite3
import json
from datetime import datetime

import httpx
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
API_BASE  = "http://localhost:8000"
DB_PATH   = os.path.join(BASE_DIR, "data", "defect_history.db")

# ─── Custom CSS (dark-industrial theme) ──────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & body ── */
:root {
    --bg-base:    #08111f;
    --bg-card:    #0d1e33;
    --bg-card2:   #112240;
    --border:     #1e3a5f;
    --border-hi:  #2a5080;
    --text-pri:   #e2edf5;
    --text-sec:   #7da0c0;
    --text-dim:   #3a607a;
    --accent:     #38bdf8;
    --accent-glow:#0ea5e9;
    --green:      #22d3a0;
    --red:        #f04255;
    --amber:      #fbbf24;
    --font:       'Inter', system-ui, sans-serif;
    --mono:       'JetBrains Mono', monospace;
}

.gradio-container, body, html { background: var(--bg-base) !important; font-family: var(--font) !important; }

/* ── Header banner ── */
.ag-header {
    background: linear-gradient(135deg, #08111f 0%, #0d1e33 40%, #0a1929 100%);
    border-bottom: 1px solid var(--border);
    padding: 24px 32px 20px;
    position: relative;
    overflow: hidden;
}
.ag-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.ag-header-title {
    font-size: 26px; font-weight: 800;
    color: var(--text-pri); letter-spacing: -0.5px;
    margin: 0 0 4px;
}
.ag-header-title span { color: var(--accent); }
.ag-header-sub {
    font-size: 13px; color: var(--text-sec);
    margin: 0; letter-spacing: 0.3px;
}
.ag-badge {
    display: inline-block; padding: 3px 10px;
    background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3);
    border-radius: 20px; font-size: 11px; color: var(--accent);
    font-weight: 600; letter-spacing: 0.5px; margin-right: 6px;
    text-transform: uppercase;
}

/* ── Tabs ── */
.tab-nav { border-bottom: 1px solid var(--border) !important; background: var(--bg-card) !important; }
.tab-nav button {
    font-family: var(--font) !important; font-size: 13px !important;
    font-weight: 500 !important; color: var(--text-sec) !important;
    padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: rgba(56,189,248,0.06) !important;
}
.tab-nav button:hover { color: var(--text-pri) !important; }

/* ── Upload area ── */
.upload-area .wrap { background: var(--bg-card) !important; border: 2px dashed var(--border-hi) !important; border-radius: 12px !important; }
.upload-area .wrap:hover { border-color: var(--accent) !important; background: rgba(56,189,248,0.04) !important; }

/* ── Buttons ── */
.btn-inspect {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 14px !important;
    color: white !important; padding: 12px 24px !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.35) !important;
    transition: all 0.2s !important;
}
.btn-inspect:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(14,165,233,0.45) !important; }

.btn-secondary {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important;
    font-weight: 500 !important; font-size: 13px !important;
    color: var(--text-sec) !important; padding: 10px 18px !important;
    transition: all 0.2s !important;
}
.btn-secondary:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── Result panel ── */
.result-panel { min-height: 420px; }

/* ── DataTable ── */
.ag-table table { width: 100% !important; }
.ag-table th {
    background: var(--bg-card2) !important; color: var(--text-sec) !important;
    font-size: 11px !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.8px !important;
    padding: 10px 14px !important; border-bottom: 1px solid var(--border) !important;
}
.ag-table td {
    background: var(--bg-card) !important; color: var(--text-pri) !important;
    font-size: 13px !important; padding: 10px 14px !important;
    border-bottom: 1px solid var(--border) !important;
}
.ag-table tr:hover td { background: var(--bg-card2) !important; }

/* ── Search box ── */
.search-box input {
    background: var(--bg-card) !important; border: 1px solid var(--border-hi) !important;
    color: var(--text-pri) !important; border-radius: 8px !important;
    font-family: var(--font) !important; font-size: 13px !important;
}
.search-box input:focus { border-color: var(--accent) !important; }

/* ── Textbox / output ── */
textarea, .output-text textarea {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-pri) !important; font-family: var(--mono) !important;
    font-size: 12px !important; border-radius: 8px !important;
}

/* ── Sections ── */
.section-label {
    font-size: 11px !important; font-weight: 600 !important;
    color: var(--text-dim) !important; text-transform: uppercase !important;
    letter-spacing: 1px !important; margin-bottom: 8px !important;
}

/* ── Accordion ── */
.gr-accordion { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* ── Footer ── */
.ag-footer {
    text-align: center; padding: 16px;
    font-size: 11px; color: var(--text-dim);
    border-top: 1px solid var(--border);
    margin-top: 8px;
}
"""

# ─── HTML Builders ────────────────────────────────────────────────────────────

def _pill(label: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 10px;'
        f'background:rgba({color},0.12);border:1px solid rgba({color},0.35);'
        f'border-radius:20px;font-size:11px;font-weight:600;color:rgb({color});'
        f'letter-spacing:0.5px;text-transform:uppercase;">{label}</span>'
    )


def build_result_html(data: dict) -> str:
    status         = data.get("status", "Unknown")
    detection      = data.get("detection", {})
    agent          = data.get("agent_analysis")
    meta           = data.get("metadata", {})

    is_pass        = status == "Pass"
    status_color   = "34,211,160" if is_pass else "240,66,85"
    status_icon    = "✅" if is_pass else "❌"
    status_label   = "PART ACCEPTED" if is_pass else "PART REJECTED"

    score          = detection.get("anomaly_score", 0.0)
    proc_time      = meta.get("processing_time_seconds", 0.0)
    latency_alert  = meta.get("latency_alert", False)
    image_file     = meta.get("image_file", "—")

    # Anomaly score bar (range ~-0.6 to 0.2 for IsolationForest)
    score_pct      = max(0, min(100, int((score + 0.6) / 0.8 * 100)))
    score_color    = "34,211,160" if score >= 0 else "240,66,85"

    # Agent section
    agent_html = ""
    if agent:
        sev      = agent.get("severity_score", 0.0)
        sev_pct  = int(sev * 100)
        if sev > 0.7:
            sev_rgb = "240,66,85"
        elif sev > 0.4:
            sev_rgb = "251,191,36"
        else:
            sev_rgb = "34,211,160"

        confirmed_badge = _pill("CONFIRMED", "240,66,85") if agent.get("defect_confirmed") else _pill("UNCONFIRMED", "251,191,36")

        agent_html = f"""
        <div style="margin-top:18px;padding:18px;
                    background:linear-gradient(135deg,rgba(240,66,85,0.06),rgba(240,66,85,0.02));
                    border:1px solid rgba(240,66,85,0.2);border-radius:10px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
                <span style="font-size:16px;">🤖</span>
                <span style="font-size:13px;font-weight:700;color:#f8d0d5;
                             text-transform:uppercase;letter-spacing:0.8px;">Agent Analysis Report</span>
                {confirmed_badge}
            </div>

            <!-- Severity -->
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:12px;color:#7da0c0;font-weight:500;">Severity Score</span>
                    <span style="font-size:14px;font-weight:700;color:rgb({sev_rgb});">{sev:.2f} / 1.00</span>
                </div>
                <div style="height:6px;background:#0d1e33;border-radius:3px;overflow:hidden;">
                    <div style="height:100%;width:{sev_pct}%;background:linear-gradient(90deg,rgb({sev_rgb}),rgba({sev_rgb},0.6));
                                border-radius:3px;transition:width 0.8s ease;"></div>
                </div>
            </div>

            <!-- Historical analogy -->
            <div style="padding:10px 12px;background:rgba(13,30,51,0.8);
                        border-left:3px solid #2a5080;border-radius:4px;margin-bottom:10px;">
                <div style="font-size:11px;color:#3a607a;font-weight:600;
                             text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">
                    Historical Analogy
                </div>
                <div style="font-size:13px;color:#b0cfe0;line-height:1.6;">
                    {agent.get('historical_analogy','—')}
                </div>
            </div>

            <!-- Recommended action -->
            <div style="padding:10px 12px;background:rgba(251,191,36,0.06);
                        border-left:3px solid #fbbf24;border-radius:4px;">
                <div style="font-size:11px;color:#7a6a20;font-weight:600;
                             text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">
                    ⚡ Recommended Action
                </div>
                <div style="font-size:13px;color:#fde68a;font-weight:500;line-height:1.6;">
                    {agent.get('recommended_action','—')}
                </div>
            </div>
        </div>
        """

    # Latency alert
    latency_html = ""
    if latency_alert:
        latency_html = """
        <div style="margin-top:8px;padding:8px 12px;
                    background:rgba(240,66,85,0.1);border-left:3px solid #f04255;border-radius:4px;">
            <span style="font-size:12px;color:#f87171;font-weight:600;">
                ⚠️ LATENCY ALERT — Processing time exceeded 500ms safety threshold
            </span>
        </div>"""

    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;color:#e2edf5;padding:4px;">

        <!-- Status Banner -->
        <div style="padding:22px 20px;
                    background:linear-gradient(135deg,rgba({status_color},0.12),rgba({status_color},0.04));
                    border:2px solid rgba({status_color},0.4);border-radius:12px;
                    text-align:center;margin-bottom:16px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                        background:radial-gradient(circle,rgba({status_color},0.15),transparent);
                        border-radius:50%;"></div>
            <div style="font-size:38px;margin-bottom:8px;">{status_icon}</div>
            <div style="font-size:24px;font-weight:800;color:rgb({status_color});
                        letter-spacing:2px;">{status_label}</div>
            <div style="font-size:12px;color:rgba({status_color},0.7);margin-top:4px;font-weight:500;">
                {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
            </div>
        </div>

        <!-- Metric Cards -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
            <!-- Anomaly Score -->
            <div style="padding:14px;background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;">
                <div style="font-size:10px;color:#3a607a;text-transform:uppercase;
                             letter-spacing:1px;margin-bottom:4px;font-weight:600;">Anomaly Score</div>
                <div style="font-size:22px;font-weight:700;color:rgb({score_color});
                             margin-bottom:6px;">{score:+.4f}</div>
                <div style="height:4px;background:#08111f;border-radius:2px;overflow:hidden;">
                    <div style="height:100%;width:{score_pct}%;background:rgb({score_color});
                                border-radius:2px;"></div>
                </div>
                <div style="font-size:10px;color:#3a607a;margin-top:4px;">More negative = more anomalous</div>
            </div>
            <!-- Processing Time -->
            <div style="padding:14px;background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;">
                <div style="font-size:10px;color:#3a607a;text-transform:uppercase;
                             letter-spacing:1px;margin-bottom:4px;font-weight:600;">Processing Time</div>
                <div style="font-size:22px;font-weight:700;
                             color:{'rgb(240,66,85)' if latency_alert else 'rgb(34,211,160)'};
                             margin-bottom:6px;">{proc_time:.3f}s</div>
                <div style="height:4px;background:#08111f;border-radius:2px;overflow:hidden;">
                    <div style="height:100%;width:{min(100,int(proc_time/0.5*100))}%;
                                background:{'rgb(240,66,85)' if latency_alert else 'rgb(34,211,160)'};
                                border-radius:2px;"></div>
                </div>
                <div style="font-size:10px;color:#3a607a;margin-top:4px;">Threshold: 0.500s</div>
            </div>
        </div>

        <!-- Image file -->
        <div style="padding:8px 12px;background:#0d1e33;border:1px solid #1e3a5f;
                    border-radius:6px;margin-bottom:8px;">
            <span style="font-size:11px;color:#3a607a;margin-right:6px;">📁 Saved as:</span>
            <span style="font-size:12px;color:#7da0c0;font-family:'JetBrains Mono',monospace;">{image_file}</span>
        </div>

        {latency_html}
        {agent_html}
    </div>
    """


def build_empty_html() -> str:
    return """
    <div style="font-family:'Inter',system-ui,sans-serif;
                display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:360px;color:#3a607a;text-align:center;padding:20px;">
        <div style="font-size:56px;margin-bottom:16px;opacity:0.4;">🔍</div>
        <div style="font-size:16px;font-weight:600;color:#1e3a5f;margin-bottom:8px;">
            Ready for Inspection
        </div>
        <div style="font-size:13px;line-height:1.6;max-width:300px;">
            Upload a part image and click<br>
            <strong style="color:#38bdf8;">Inspect Part</strong> to run the AI pipeline
        </div>
    </div>
    """


def build_error_html(title: str, message: str, code: str = "") -> str:
    code_block = f"""
    <div style="margin-top:10px;padding:10px;background:#08111f;border-radius:6px;
                font-family:'JetBrains Mono',monospace;font-size:12px;color:#7da0c0;">
        {code}
    </div>""" if code else ""

    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;padding:20px;
                background:rgba(240,66,85,0.06);border:1px solid rgba(240,66,85,0.25);
                border-radius:10px;color:#f87171;">
        <div style="font-size:15px;font-weight:700;margin-bottom:8px;">⚠️ {title}</div>
        <div style="font-size:13px;color:#fca5a5;line-height:1.6;">{message}</div>
        {code_block}
    </div>
    """


# ─── Backend Functions ────────────────────────────────────────────────────────

def inspect_part(image_path):
    """POST the uploaded image to FastAPI /inspect-part and render results."""
    if image_path is None:
        return build_empty_html(), ""

    try:
        with open(image_path, "rb") as f:
            ext      = os.path.splitext(image_path)[1] or ".jpg"
            mime     = "image/png" if ext.lower() == ".png" else "image/jpeg"
            response = httpx.post(
                f"{API_BASE}/inspect-part",
                files={"file": (f"part{ext}", f, mime)},
                timeout=120.0
            )

        if response.status_code == 200:
            data     = response.json()
            raw_json = json.dumps(data, indent=2)
            return build_result_html(data), raw_json

        elif response.status_code == 503:
            detail = response.json().get("detail", "Service unavailable")
            return build_error_html("Service Unavailable", detail), ""

        else:
            return build_error_html(
                f"HTTP {response.status_code}",
                response.text[:500]
            ), ""

    except httpx.ConnectError:
        return build_error_html(
            "Backend Offline",
            "Cannot reach the API server at <code>http://localhost:8000</code>. "
            "Start it with:",
            "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1"
        ), ""

    except FileNotFoundError:
        return build_error_html("File Error", "Uploaded image not found on disk."), ""

    except Exception as exc:
        return build_error_html("Unexpected Error", str(exc)), ""


def load_defect_history(search_term: str = ""):
    """Read defect history directly from SQLite — no HTTP hop needed."""
    # Ensure DB is seeded if it doesn't exist yet
    if not os.path.exists(DB_PATH):
        try:
            sys.path.insert(0, BASE_DIR)
            from src.agent import _initialize_database
            _initialize_database()
        except Exception:
            return pd.DataFrame({"Info": ["Database not yet created. Run an inspection first."]})

    try:
        conn   = sqlite3.connect(os.path.abspath(DB_PATH))
        term   = f"%{search_term.strip()}%"
        query  = """
            SELECT
                id          AS "ID",
                quarter     AS "Quarter",
                defect_signature AS "Defect Signature",
                ROUND(severity, 3) AS "Severity",
                description AS "Description",
                resolution  AS "Resolution"
            FROM defect_history
            WHERE (? = '%%'
                   OR defect_signature LIKE ?
                   OR description      LIKE ?
                   OR resolution       LIKE ?
                   OR quarter          LIKE ?)
            ORDER BY severity DESC
        """
        df = pd.read_sql_query(query, conn, params=(term, term, term, term, term))
        conn.close()
        return df if not df.empty else pd.DataFrame({"Info": ["No records match your search."]})
    except Exception as exc:
        return pd.DataFrame({"Error": [str(exc)]})


def get_defect_stats():
    """Summary statistics card from the SQLite DB."""
    if not os.path.exists(DB_PATH):
        return _stats_html(0, 0.0, "—")
    try:
        conn   = sqlite3.connect(os.path.abspath(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(severity), MAX(severity) FROM defect_history")
        row    = cursor.fetchone()
        conn.close()
        count, avg_sev, max_sev = row
        return _stats_html(count or 0, avg_sev or 0.0, f"{max_sev:.2f}" if max_sev else "—")
    except Exception as exc:
        return f"<div style='color:#ef4444;font-family:system-ui;'>Error: {exc}</div>"


def _stats_html(count, avg, max_sev):
    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;
                display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:4px;">
        {_stat_card("Total Records", str(count), "📋")}
        {_stat_card("Avg Severity", f"{avg:.3f}", "📊")}
        {_stat_card("Max Severity", str(max_sev), "🔴")}
    </div>
    """


def _stat_card(label, value, icon):
    return f"""
    <div style="padding:14px;background:#0d1e33;border:1px solid #1e3a5f;
                border-radius:8px;text-align:center;">
        <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
        <div style="font-size:20px;font-weight:700;color:#38bdf8;">{value}</div>
        <div style="font-size:11px;color:#3a607a;text-transform:uppercase;
                     letter-spacing:0.8px;margin-top:4px;">{label}</div>
    </div>
    """


def check_health():
    """GET /health from FastAPI and render a status card."""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
        if resp.status_code == 200:
            d = resp.json()
            return _health_html(
                api_ok    = d.get("status") == "healthy",
                enc_ok    = d.get("encoder_loaded", False),
                det_ok    = d.get("detector_loaded", False),
                train_ok  = d.get("detector_trained", False),
            )
        return build_error_html(f"HTTP {resp.status_code}", resp.text[:300])
    except httpx.ConnectError:
        return build_error_html(
            "API Server Offline",
            "Cannot connect to <code>http://localhost:8000</code>.",
            "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1"
        )
    except Exception as exc:
        return build_error_html("Health Check Failed", str(exc))


def _health_html(api_ok, enc_ok, det_ok, train_ok):
    def row(label, ok, note=""):
        color = "34,211,160" if ok else "240,66,85"
        dot   = "●" if ok else "○"
        state = "ONLINE" if ok else "OFFLINE"
        note_span = f'<span style="font-size:11px;color:#3a607a;margin-left:6px;">({note})</span>' if note else ""
        return f"""
        <tr>
            <td style="padding:11px 14px;color:#7da0c0;font-size:13px;">{label}</td>
            <td style="padding:11px 14px;">
                <span style="color:rgb({color});font-weight:600;font-size:13px;">{dot} {state}</span>
                {note_span}
            </td>
        </tr>"""

    overall_color = "34,211,160" if all([api_ok, enc_ok, det_ok, train_ok]) else "240,66,85"
    overall_label = "ALL SYSTEMS OPERATIONAL" if all([api_ok, enc_ok, det_ok, train_ok]) else "DEGRADED"

    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;color:#e2edf5;">
        <!-- Overall status -->
        <div style="padding:14px 18px;margin-bottom:14px;
                    background:rgba({overall_color},0.08);
                    border:1px solid rgba({overall_color},0.3);border-radius:10px;
                    display:flex;align-items:center;gap:10px;">
            <span style="font-size:20px;">{'✅' if 'OPERATIONAL' in overall_label else '⚠️'}</span>
            <span style="font-size:14px;font-weight:700;color:rgb({overall_color});
                         letter-spacing:0.5px;">{overall_label}</span>
        </div>
        <!-- Status table -->
        <table style="width:100%;border-collapse:collapse;
                      background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;overflow:hidden;">
            <thead>
                <tr style="background:#112240;border-bottom:1px solid #1e3a5f;">
                    <th style="padding:9px 14px;text-align:left;font-size:11px;
                                color:#3a607a;text-transform:uppercase;letter-spacing:0.8px;
                                font-weight:600;">Component</th>
                    <th style="padding:9px 14px;text-align:left;font-size:11px;
                                color:#3a607a;text-transform:uppercase;letter-spacing:0.8px;
                                font-weight:600;">Status</th>
                </tr>
            </thead>
            <tbody>
                {row('API Server', api_ok)}
                {row('VisionEncoder', enc_ok, 'ResNet-18, CPU-only')}
                {row('AnomalyDetector', det_ok, 'IsolationForest')}
                {row('Detector Trained', train_ok, 'model weights loaded')}
            </tbody>
        </table>
        <div style="font-size:11px;color:#3a607a;margin-top:10px;text-align:right;">
            Last checked: {datetime.now().strftime('%H:%M:%S')}
        </div>
    </div>
    """


def get_prometheus_metrics():
    """Fetch and format raw Prometheus metrics."""
    try:
        resp = httpx.get(f"{API_BASE}/metrics/", timeout=5.0, follow_redirects=True)
        if resp.status_code == 200:
            lines = [l for l in resp.text.splitlines()
                     if "inspection_latency" in l or l.startswith("# HELP") or l.startswith("# TYPE")]
            return "\n".join(lines) if lines else "No metrics recorded yet — run some inspections."
        return f"HTTP {resp.status_code}: {resp.text[:200]}"
    except httpx.ConnectError:
        return "⚠️  Backend offline — metrics unavailable."
    except Exception as exc:
        return f"Error: {exc}"


# ─── Gradio App ───────────────────────────────────────────────────────────────

PIPELINE_MD = """
## How the Pipeline Works

```
Upload Image  →  VisionEncoder (ResNet-18)  →  AnomalyDetector (IsolationForest)
                   512-D feature vector            Normal / Anomaly + score
                                                        ↙          ↘
                                                  ✅ PASS       ❌ FAIL
                                              (Agent bypassed) (LLM Agent activated)
                                                                      ↓
                                                         Query SQLite Defect DB
                                                                      ↓
                                                         Groq LLaMA 3.3 70B
                                                                      ↓
                                                    Structured JSON Rejection Report
```

### Components

| Module | Technology | Role |
|--------|-----------|------|
| `src/encoder.py` | ResNet-18 (PyTorch) | Extracts 512-D L2-normalized image embeddings |
| `src/detector.py` | IsolationForest (sklearn) | Unsupervised outlier detection on embeddings |
| `src/agent.py` | LangChain + Groq LLaMA 3.3 70B | Generates structured rejection reports |
| `src/api/main.py` | FastAPI + Uvicorn | REST API serving the inspection pipeline |
| `data/defect_history.db` | SQLite | Local historical defect reference database |
| `app.py` | Gradio | This interactive dashboard |

### Design Principles
- **Zero-label training** — IsolationForest learns only from *good* parts; no defect annotations needed
- **Cost-efficient LLM usage** — the agent only activates on anomalies (~5% of inspections)
- **Edge-first** — fully CPU-bound, single Uvicorn worker, ≤8GB RAM footprint
- **500ms latency budget** — hard threshold aligned with assembly line throughput requirements
"""

with gr.Blocks(
    title="Agentic Vision — Industrial QC",
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="ag-header">
        <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
            <div style="font-size:36px;">🏭</div>
            <div>
                <h1 class="ag-header-title">Agentic <span>Vision</span></h1>
                <p class="ag-header-sub">AI-powered defect detection for industrial assembly lines</p>
            </div>
            <div style="margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;">
                <span class="ag-badge">ResNet-18</span>
                <span class="ag-badge">IsolationForest</span>
                <span class="ag-badge">LLaMA 3.3 70B</span>
                <span class="ag-badge">FastAPI</span>
            </div>
        </div>
    </div>
    """)

    # ── Tabs ────────────────────────────────────────────────────────────────
    with gr.Tabs(elem_classes="tab-nav"):

        # ════════════════════════════════════════════════════════════════════
        # TAB 1 — PART INSPECTION
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("🔍  Part Inspection"):
            with gr.Row(equal_height=False):

                # Left column — upload
                with gr.Column(scale=4, min_width=300):
                    gr.HTML('<div class="section-label" style="margin-top:12px;">Upload Part Image</div>')
                    img_input = gr.Image(
                        label="",
                        type="filepath",
                        elem_classes="upload-area",
                        show_label=False,
                        height=280,
                    )
                    inspect_btn = gr.Button(
                        "🔍  Inspect Part",
                        variant="primary",
                        elem_classes="btn-inspect",
                        size="lg",
                    )
                    gr.HTML('<div style="margin-top:10px;font-size:12px;color:#3a607a;line-height:1.6;">'
                            '📤 Supported formats: JPG, PNG, BMP, TIFF<br>'
                            '⚡ Processing typically completes in &lt;300ms<br>'
                            '🤖 LLM agent activates only when a defect is flagged'
                            '</div>')

                # Right column — results
                with gr.Column(scale=6, min_width=380):
                    gr.HTML('<div class="section-label" style="margin-top:12px;">Inspection Results</div>')
                    result_html = gr.HTML(
                        value=build_empty_html(),
                        elem_classes="result-panel",
                    )
                    with gr.Accordion("📄 Raw JSON Response", open=False):
                        raw_json_out = gr.Code(
                            language="json",
                            label="",
                            interactive=False,
                        )

            inspect_btn.click(
                fn=inspect_part,
                inputs=[img_input],
                outputs=[result_html, raw_json_out],
                show_progress=True,
            )

        # ════════════════════════════════════════════════════════════════════
        # TAB 2 — DEFECT HISTORY
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("📚  Defect History"):
            gr.HTML('<div style="height:12px;"></div>')

            # Stats row
            stats_html = gr.HTML(value=get_defect_stats())

            gr.HTML('<div style="height:12px;"></div>')
            gr.HTML('<div class="section-label">Search Historical Records</div>')

            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="Search by defect type, description, resolution…",
                    show_label=False,
                    elem_classes="search-box",
                    scale=5,
                )
                search_btn = gr.Button("🔍  Search", elem_classes="btn-secondary", scale=1)
                refresh_hist_btn = gr.Button("🔄  Reload All", elem_classes="btn-secondary", scale=1)

            gr.HTML('<div style="height:8px;"></div>')
            history_table = gr.DataFrame(
                value=load_defect_history(),
                label="",
                wrap=True,
                elem_classes="ag-table",
            )

            search_btn.click(fn=load_defect_history, inputs=[search_box], outputs=[history_table])
            search_box.submit(fn=load_defect_history, inputs=[search_box], outputs=[history_table])
            refresh_hist_btn.click(
                fn=lambda: (load_defect_history(), get_defect_stats()),
                inputs=[],
                outputs=[history_table, stats_html],
            )

        # ════════════════════════════════════════════════════════════════════
        # TAB 3 — SYSTEM HEALTH
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("📊  System Health"):
            gr.HTML('<div style="height:12px;"></div>')

            with gr.Row():
                refresh_health_btn = gr.Button(
                    "🔄  Refresh Status", elem_classes="btn-secondary", scale=1
                )
                gr.HTML('<div style="flex:1;"></div>')

            gr.HTML('<div class="section-label" style="margin-top:12px;">Component Status</div>')
            health_html = gr.HTML(value=check_health())

            gr.HTML('<div class="section-label" style="margin-top:16px;">Prometheus Metrics</div>')
            metrics_out = gr.Code(
                value=get_prometheus_metrics(),
                label="",
                interactive=False,
            )

            refresh_health_btn.click(
                fn=lambda: (check_health(), get_prometheus_metrics()),
                inputs=[],
                outputs=[health_html, metrics_out],
            )

        # ════════════════════════════════════════════════════════════════════
        # TAB 4 — ABOUT / PIPELINE
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("ℹ️  About"):
            gr.HTML('<div style="height:12px;"></div>')
            gr.Markdown(PIPELINE_MD)
            gr.HTML("""
            <div style="margin-top:20px;padding:16px;background:#0d1e33;
                        border:1px solid #1e3a5f;border-radius:8px;
                        font-family:'Inter',system-ui,sans-serif;">
                <div style="font-size:13px;font-weight:600;color:#38bdf8;margin-bottom:10px;">
                    📡 API Endpoints
                </div>
                <div style="display:grid;gap:8px;">
                    <div style="display:flex;gap:10px;align-items:center;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;
                                     padding:3px 8px;background:#112240;border-radius:4px;
                                     color:#fbbf24;">POST</span>
                        <code style="font-size:12px;color:#7da0c0;">/inspect-part</code>
                        <span style="font-size:12px;color:#3a607a;">— Upload image for inspection</span>
                    </div>
                    <div style="display:flex;gap:10px;align-items:center;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;
                                     padding:3px 8px;background:#112240;border-radius:4px;
                                     color:#22d3a0;">GET</span>
                        <code style="font-size:12px;color:#7da0c0;">/health</code>
                        <span style="font-size:12px;color:#3a607a;">— Model readiness probe</span>
                    </div>
                    <div style="display:flex;gap:10px;align-items:center;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;
                                     padding:3px 8px;background:#112240;border-radius:4px;
                                     color:#22d3a0;">GET</span>
                        <code style="font-size:12px;color:#7da0c0;">/metrics</code>
                        <span style="font-size:12px;color:#3a607a;">— Prometheus latency metrics</span>
                    </div>
                    <div style="display:flex;gap:10px;align-items:center;">
                        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;
                                     padding:3px 8px;background:#112240;border-radius:4px;
                                     color:#22d3a0;">GET</span>
                        <code style="font-size:12px;color:#7da0c0;">/docs</code>
                        <span style="font-size:12px;color:#3a607a;">— Interactive Swagger UI</span>
                    </div>
                </div>
            </div>
            """)

    # ── Footer ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="ag-footer">
        Agentic Vision for Industrial Quality Control &nbsp;•&nbsp;
        ResNet-18 + IsolationForest + LLaMA 3.3 70B &nbsp;•&nbsp;
        <a href="https://github.com/atharvadarke/Agentic-Vision-for-Industrial-Quality-Control"
           target="_blank" style="color:#38bdf8;text-decoration:none;">GitHub ↗</a>
    </div>
    """)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic Vision — Gradio Dashboard")
    print("=" * 60)
    print(f"  API backend expected at: {API_BASE}")
    print(f"  SQLite DB path:          {DB_PATH}")
    print("  Dashboard launching at:  http://localhost:7860")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(primary_hue="sky", neutral_hue="slate"),
        css=CSS,
    )
