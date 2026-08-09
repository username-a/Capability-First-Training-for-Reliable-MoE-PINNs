"""
Multi-stream real-time dashboard.

Reads status files written by the experiment schedulers:
    - Burger2D/results/extended_status.json            (2D balance sweep + K sensitivity)
    - burger1D/results/gate_intro_1d_status.json       (1D gate introduction continuum)
    - ShallowWater2D/results/dam_break_gate_intro_status.json (dam-break gate continuum)

Endpoints: /  /api/status  /api/rows  /chart.png
Usage:
    python Burger2D/scripts/live_dashboard.py [--port 8765]
"""

from __future__ import annotations

import argparse
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

STREAMS = [
    ("2D Extended (balance/K)", os.path.join(PACKAGE_ROOT, "results", "extended_status.json"), "#4C72B0"),
    ("KdV Gate-Intro", os.path.join(PROJECT_ROOT, "burger1D", "results", "kdv_gate_intro_status.json"), "#C44E52"),
]


def _read_status(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _rows() -> list[dict]:
    rows = []
    for name, path, color in STREAMS:
        st = _read_status(path)
        if not st:
            continue
        done = st.get("done", {})
        for tag, entry in done.items():
            if "error" in entry:
                rows.append({"stream": name, "tag": tag, "status": "失败", "l2": None,
                             "extra": None, "color": color})
                continue
            if name.startswith("2D"):
                m = entry.get("metrics", {})
                l2 = m.get("l2_relative_error")
                extra = m.get("max_absolute_error")
            elif name.startswith("1D") or name.startswith("KdV"):
                l2 = entry.get("l2")
                extra = entry.get("maxerr")
            else:  # dam
                m = entry.get("metrics", {})
                l2 = m.get("l2_rel_mean")
                extra = m.get("shock_position_error")
            rows.append({"stream": name, "tag": tag, "status": "完成", "l2": l2,
                         "extra": extra, "color": color})
        cur = st.get("current")
        if cur:
            cur_tag = cur.get("tag") if isinstance(cur, dict) else cur
            rows.append({"stream": name, "tag": cur_tag, "status": "训练中",
                         "l2": None, "extra": None, "color": color})
    return rows


def _status() -> dict:
    streams = []
    for name, path, color in STREAMS:
        st = _read_status(path)
        if not st:
            streams.append({"name": name, "total": 0, "completed": 0, "current": None,
                            "exists": False, "color": color})
            continue
        cur = st.get("current")
        cur_tag = cur.get("tag") if isinstance(cur, dict) else cur
        streams.append({
            "name": name,
            "total": st.get("total", 0),
            "completed": st.get("completed", len(st.get("done", {}))),
            "current": cur_tag,
            "exists": True,
            "color": color,
        })
    return {"streams": streams}


def _chart_png(rows: list[dict]) -> bytes:
    names = [s[0] for s in STREAMS]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, name in zip(axes, names):
        stream_rows = [r for r in rows if r["stream"] == name and r["status"] == "完成"]
        if not stream_rows:
            ax.set_title(name, fontsize=10)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888")
            continue
        tags = [r["tag"] for r in stream_rows]
        l2 = [r["l2"] for r in stream_rows]
        xs = range(len(tags))
        color = stream_rows[0]["color"]
        ax.plot(list(xs), l2, marker="o", ms=4, lw=1.8, color=color)
        for x, t, v in zip(xs, tags, l2):
            ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7)
        if name.startswith("KdV"):
            extra = [r["extra"] for r in stream_rows]
            ax.plot(list(xs), extra, marker="s", ms=4, lw=1.4, ls="--", color="#55A868",
                    label="MaxErr")
            ax.legend(fontsize=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([t.replace("_seed", "\n") for t in tags], fontsize=6, rotation=0)
        ax.set_title(f"{name}\n(L2 per completed run)", fontsize=10)
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
    fig.suptitle("Experiment streams - L2 per completed run", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send(json.dumps(_status(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif parsed.path == "/api/rows":
            self._send(json.dumps(_rows(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif parsed.path == "/chart.png":
            self._send(_chart_png(_rows()), "image/png")
        else:
            self._send(_PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # noqa: N802
        print(f"[dashboard] {fmt % args}", flush=True)


_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>MoE-PINN 实验看板</title>
<style>
:root{--bg:#0b1220;--panel:#121c30;--panel2:#182539;--line:#233149;--txt:#e6edf7;
--muted:#8fa3bf;--ok:#34d399;--warn:#fbbf24;--idle:#64748b;}
*{box-sizing:border-box}body{margin:0;font-family:"Segoe UI","Microsoft YaHei",sans-serif;
background:radial-gradient(1200px 600px at 20% -10%,#16233d 0%,var(--bg) 55%);color:var(--txt);
min-height:100vh;padding:22px 26px 40px;}
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap;gap:10px}
h1{font-size:21px;margin:0;font-weight:650}h1 small{color:var(--muted);font-weight:400;font-size:13px;margin-left:10px}
.live{display:inline-flex;align-items:center;gap:7px;color:var(--ok);font-size:13px;font-weight:600;
background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.35);padding:5px 12px;border-radius:999px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);animation:pulse 1.6s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.streams{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}
.scard{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
border-radius:14px;padding:14px 16px}
.scard .k{color:var(--muted);font-size:12px;margin-bottom:8px}
.scard .v{font-size:22px;font-weight:700}
.scard .v small{font-size:13px;color:var(--muted);font-weight:400}
.scard .cur{color:var(--warn);font-size:12px;margin-top:6px;word-break:break-all}
.section{margin-bottom:14px}.section h2{font-size:14px;color:var(--muted);font-weight:600;margin:0 0 10px 2px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:14px}
.panel img{width:100%;border-radius:10px;display:block}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{color:var(--muted);font-weight:600;text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
td{padding:7px 8px;border-bottom:1px solid #182539;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.badge{font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px}
.badge.done{color:var(--ok);background:rgba(52,211,153,.12)}
.badge.running{color:var(--warn);background:rgba(251,191,36,.12)}
.badge.fail{color:#ff6b6b;background:rgba(255,107,107,.12)}
@media(max-width:1000px){.streams{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header"><h1>MoE-PINN 实验实时看板<small>2D 扩展 · 1D · Dam-break</small></h1>
<span class="live"><span class="dot"></span>LIVE</span></div>
<div class="streams" id="streams"></div>
<div class="panel"><img id="chart" src="/chart.png" alt="trajectories"></div>
<div class="section"><h2>RUNS</h2></div>
<div class="panel"><table id="tbl"></table></div>
<script>
async function refresh(){
 try{
  const st=await(await fetch('/api/status')).json();
  let cards='';
  for(const s of st.streams){
   cards+=`<div class="scard"><div class="k">${s.name}${s.exists?'':'（未启动）'}</div>
    <div class="v">${s.completed}<small>/${s.total}</small></div>
    <div class="cur">${s.current?('当前: '+s.current):(s.exists?'等待/完成':'')}</div></div>`;
  }
  document.getElementById('streams').innerHTML=cards;
  const rows=await(await fetch('/api/rows')).json();
  let html='<tr><th>stream</th><th>run</th><th>状态</th><th>L2</th><th>辅助指标</th></tr>';
  for(const r of rows){
   const cls=r.status==='完成'?'done':(r.status==='训练中'?'running':'fail');
   html+=`<tr><td>${r.stream}</td><td style="font-family:Consolas,monospace">${r.tag}</td>
    <td><span class="badge ${cls}">${r.status}</span></td>
    <td>${r.l2===null||r.l2===undefined?'-':Number(r.l2).toFixed(4)}</td>
    <td>${r.extra===null||r.extra===undefined?'-':Number(r.extra).toFixed(4)}</td></tr>`;
  }
  document.getElementById('tbl').innerHTML=html;
  document.getElementById('chart').src='/chart.png?t='+Date.now();
 }catch(e){}
}
refresh();setInterval(refresh,3000);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[OK] Dashboard: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
