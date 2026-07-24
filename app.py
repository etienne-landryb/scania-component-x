"""
app.py -- Component X fleet triage.

A maintenance planner's worklist, not a model dashboard. The question it answers
is "which trucks do I bring in this week?", and the answer is ranked by money
saved rather than by predicted probability.

Everything on screen recomputes live from two controls:
  * inspection capacity  -- how many trucks the workshop can actually take
  * cost of a missed failure -- the economic assumption behind every decision

The model's probabilities are precomputed (see export_demo.py); the DECISION
layer is fully live, which is where this project's contribution sits.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo")

st.set_page_config(page_title="Component X — Fleet Triage",
                   layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Design tokens
#   Type: IBM Plex, drawn for engineering interfaces. Condensed for signage-like
#   headers, Mono for every number so columns align like a printed worklist.
#
#   Colour is deliberately theme-agnostic so Streamlit's own System/Light/Dark
#   switch keeps working. Instead of fixed greys, panels and rules use
#   TRANSLUCENT neutrals -- rgba(128,138,145,a) sits slightly darker than a light
#   background and slightly lighter than a dark one, so one rule serves both.
#   Body text inherits Streamlit's text colour; muted text is opacity, not grey.
#   Only the three signal colours are fixed, chosen at a luminance that stays
#   legible on either ground.
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root{
  --urgent:#B93415; --urgent2:#8C2710; --watch:#C4820C; --clear:#3D8B5F;
  --panel:rgba(128,138,145,.10);
  --rule:rgba(128,138,145,.38);
  --hairline:rgba(128,138,145,.20);
  --track:rgba(128,138,145,.22);
  --fill:rgba(128,138,145,.70);
}

html, body, [class*="css"]{ font-family:'IBM Plex Sans',system-ui,sans-serif; }

/* ---- masthead ---- */
.mast{ border-bottom:2px solid currentColor; padding:.2rem 0 .7rem; margin-bottom:1.1rem; }
.mast h1{
  font-family:'IBM Plex Sans Condensed',sans-serif; font-weight:700;
  font-size:2.35rem; letter-spacing:.02em; text-transform:uppercase;
  margin:0; line-height:1.05; color:inherit;
}
.mast .sub{ font-family:'IBM Plex Mono',monospace; font-size:.78rem;
  opacity:.62; letter-spacing:.04em; margin-top:.35rem; }

/* ---- benchmark strip ---- */
.bench{ display:flex; gap:0; border:1px solid var(--rule); background:var(--panel);
  margin-bottom:1.4rem; flex-wrap:wrap; }
.bench .cell{ flex:1 1 150px; padding:.7rem .9rem; border-right:1px solid var(--rule); }
.bench .cell:last-child{ border-right:none; }
.bench .k{ font-family:'IBM Plex Mono',monospace; font-size:.66rem;
  letter-spacing:.09em; text-transform:uppercase; opacity:.62; }
.bench .v{ font-family:'IBM Plex Mono',monospace; font-size:1.5rem;
  font-weight:600; line-height:1.25; color:inherit; }
.bench .v.good{ color:var(--clear); }

/* ---- section headings ---- */
.sect{ font-family:'IBM Plex Sans Condensed',sans-serif; font-weight:700;
  text-transform:uppercase; letter-spacing:.05em; font-size:1.05rem;
  border-bottom:1px solid currentColor; padding-bottom:.3rem; margin:1.5rem 0 .8rem; }

/* ---- worklist ---- */
table.wl{ width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace;
  font-size:.82rem; background:var(--panel); }
table.wl th{ text-align:left; font-size:.64rem; letter-spacing:.09em;
  text-transform:uppercase; opacity:.62; font-weight:500;
  border-bottom:1px solid var(--rule); padding:.45rem .6rem; }
table.wl td{ padding:.42rem .6rem; border-bottom:1px solid var(--hairline);
  color:inherit; }
table.wl tr:last-child td{ border-bottom:none; }
.rank{ opacity:.55; }
.vid{ font-weight:600; }
.num{ text-align:right; }

/* priority chips. The hazard-stripe fill is used ONCE, on urgent only. */
.chip{ display:inline-block; padding:.1rem .5rem; font-size:.68rem;
  letter-spacing:.06em; text-transform:uppercase; font-weight:600;
  border:1px solid currentColor; }
.chip.urgent{ color:#fff; border-color:var(--urgent);
  background-image:repeating-linear-gradient(45deg,var(--urgent) 0 7px,var(--urgent2) 7px 14px); }
.chip.watch{ color:var(--watch); }
.chip.clear{ color:var(--clear); }

/* ---- decision bar : the signature element ---- */
.dec{ background:var(--panel); border:1px solid var(--rule); padding:.9rem 1rem; }
.dec .r{ display:flex; align-items:center; gap:.7rem; margin:.3rem 0; }
.dec .lbl{ font-family:'IBM Plex Mono',monospace; font-size:.74rem;
  width:12.5rem; opacity:.62; }
.dec .bar{ flex:1; height:15px; background:var(--track); position:relative; }
.dec .bar i{ position:absolute; inset:0 auto 0 0; background:var(--fill); display:block; }
.dec .r.chosen .lbl{ opacity:1; font-weight:600; }
.dec .r.chosen .bar i{ background:var(--urgent); }
.dec .val{ font-family:'IBM Plex Mono',monospace; font-size:.78rem;
  width:4.4rem; text-align:right; color:inherit; }
.disagree{ border-left:3px solid var(--urgent); padding:.55rem .8rem;
  background:rgba(185,52,21,.12); font-size:.87rem; margin-top:.7rem; color:inherit; }
.agree{ border-left:3px solid var(--rule); padding:.55rem .8rem;
  background:var(--panel); font-size:.87rem; margin-top:.7rem; color:inherit; }

.note{ font-size:.8rem; opacity:.62; }
[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono',monospace; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


CLASS_WINDOW = {
    0: "No failure signal",
    1: "24–48 units out",
    2: "12–24 units out",
    3: "6–12 units out",
    4: "0–6 units out",
}
ACTIONS = ["Keep running", "Inspect at next service", "Schedule inspection",
           "Book workshop visit", "Take in before next mission"]


@st.cache_data
def load_demo():
    v = pd.read_parquet(os.path.join(DEMO, "vehicles.parquet"))
    with open(os.path.join(DEMO, "meta.json")) as f:
        m = json.load(f)
    return v, m


def build_cost_matrix(miss_cost: int) -> np.ndarray:
    """Cost matrix with the miss penalty scaled to the planner's assumption.

    The published matrix charges 500 for calling an imminent failure healthy.
    A planner whose trucks carry costlier cargo can raise it and watch every
    decision on the page move.
    """
    base = np.array([
        [0,   7,   8,   9,  10],
        [200, 0,   7,   8,   9],
        [300, 200, 0,   7,   8],
        [400, 300, 200, 0,   7],
        [500, 400, 300, 200, 0],
    ], dtype=float)
    scale = miss_cost / 500.0
    out = base.copy()
    lower = np.tril(np.ones_like(base), -1).astype(bool)   # false negatives
    out[lower] = base[lower] * scale
    return out


try:
    vehicles, meta = load_demo()
except Exception:
    st.error("Demo data not found. Run export_demo.py locally, then deploy the "
             "demo/ folder alongside app.py.")
    st.stop()

P = vehicles[[f"p{c}" for c in range(5)]].to_numpy()
y_true = vehicles["true_class"].to_numpy()
n_fleet = len(vehicles)

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("<div class='sect' style='margin-top:0'>Planning</div>",
                unsafe_allow_html=True)
    capacity = st.slider("Inspections available this week", 1,
                         min(200, n_fleet), min(40, n_fleet))
    st.caption("The workshop cannot take every flagged truck. "
               "The worklist is ranked so the ones you can take are the ones "
               "that save the most.")

    st.markdown("<div class='sect'>Economics</div>", unsafe_allow_html=True)
    miss_cost = st.slider("Cost of missing an imminent failure", 100, 2000, 500, 50)
    st.caption("A breakdown on the road versus a workshop check. "
               "The published challenge figure is 500.")

COST = build_cost_matrix(miss_cost)
exp_cost = P @ COST                       # expected cost of each action
best_action = exp_cost.argmin(axis=1)
do_nothing = exp_cost[:, 0]
saving = do_nothing - exp_cost[np.arange(n_fleet), best_action]

order = np.argsort(-saving)
worklist = order[:capacity]
worklist = worklist[saving[worklist] > 0]

# realised cost under this capacity: inspect the top N, leave the rest alone
plan = np.zeros(n_fleet, dtype=int)
plan[worklist] = best_action[worklist]
realised = COST[y_true, plan].sum()
nothing_cost = COST[y_true, np.zeros(n_fleet, int)].sum()
caught = int(((plan == 4) & (y_true == 4)).sum())
total_urgent = int((y_true == 4).sum())

# --------------------------------------------------------------------------- #
# Masthead + benchmark
# --------------------------------------------------------------------------- #
b = meta["benchmark"]
st.markdown(
    "<div class='mast'><h1>Component X · Fleet Triage</h1>"
    "<div class='sub'>Cost-sensitive predictive maintenance · "
    f"SCANIA Component X benchmark · {b['test_vehicles']:,} held-out vehicles</div></div>",
    unsafe_allow_html=True)

st.markdown(f"""
<div class='bench'>
  <div class='cell'><div class='k'>Benchmark cost</div>
    <div class='v good'>{b['model_total_cost']:,.0f}</div></div>
  <div class='cell'><div class='k'>If nothing is done</div>
    <div class='v'>{b['naive_total_cost']:,.0f}</div></div>
  <div class='cell'><div class='k'>If everything is inspected</div>
    <div class='v'>{b['inspect_all_total_cost']:,.0f}</div></div>
  <div class='cell'><div class='k'>Cost avoided</div>
    <div class='v good'>{b['pct_below_naive']}%</div></div>
  <div class='cell'><div class='k'>Failures caught, unconstrained</div>
    <div class='v'>{b['class4_recall']*100:.0f}%</div></div>
</div>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# This week's plan
# --------------------------------------------------------------------------- #
st.markdown("<div class='sect'>This week's plan</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Trucks going in", f"{len(worklist)}",
          f"of {n_fleet} in the fleet", delta_color="off")
c2.metric("Cost with this plan", f"{realised:,.0f}",
          f"{realised - nothing_cost:,.0f} vs doing nothing", delta_color="inverse")
c3.metric("Caught within this capacity", f"{caught} / {total_urgent}",
          "unconstrained model catches 80%", delta_color="off")

rows = []
for rank, i in enumerate(worklist, start=1):
    a = int(best_action[i])
    cls = "urgent" if a == 4 else ("watch" if a > 0 else "clear")
    rows.append(
        f"<tr><td class='rank'>{rank:02d}</td>"
        f"<td class='vid'>#{int(vehicles.vehicle_id.iloc[i])}</td>"
        f"<td><span class='chip {cls}'>{ACTIONS[a]}</span></td>"
        f"<td class='num'>{P[i,4]*100:.1f}%</td>"
        f"<td class='num'>{saving[i]:,.0f}</td></tr>")

if rows:
    st.markdown(
        "<table class='wl'><tr><th>#</th><th>Vehicle</th><th>Action</th>"
        "<th class='num'>Imminent risk</th><th class='num'>Cost avoided</th></tr>"
        + "".join(rows) + "</table>", unsafe_allow_html=True)
else:
    st.markdown("<p class='note'>No truck currently justifies an inspection at "
                "these cost assumptions. Raise the cost of a missed failure to "
                "see the threshold move.</p>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Single vehicle — the signature view
# --------------------------------------------------------------------------- #
st.markdown("<div class='sect'>Why this truck</div>", unsafe_allow_html=True)

labels = [f"#{int(vehicles.vehicle_id.iloc[i])}  ·  {saving[i]:,.0f} saved"
          for i in order]
choice = st.selectbox("Vehicle", labels, index=0, label_visibility="collapsed")
sel = order[labels.index(choice)]

left, right = st.columns([1, 1])

with left:
    st.markdown("<div class='note' style='margin-bottom:.5rem'>"
                "Predicted risk profile</div>", unsafe_allow_html=True)
    prof = "".join(
        f"<div class='r'><span class='lbl'>{CLASS_WINDOW[c]}</span>"
        f"<div class='bar'><i style='width:{max(P[sel,c]*100,0.6):.1f}%'></i></div>"
        f"<span class='val'>{P[sel,c]*100:.1f}%</span></div>" for c in range(5))
    st.markdown(f"<div class='dec'>{prof}</div>", unsafe_allow_html=True)

    if "age_at_readout" in vehicles.columns:
        st.markdown(
            f"<p class='note' style='margin-top:.6rem'>"
            f"In service {vehicles.age_at_readout.iloc[sel]:,.0f} time units · "
            f"{vehicles.readouts_recorded.iloc[sel]:,.0f} readouts recorded</p>",
            unsafe_allow_html=True)

with right:
    st.markdown("<div class='note' style='margin-bottom:.5rem'>"
                "Expected cost of each action</div>", unsafe_allow_html=True)
    ec = exp_cost[sel]
    hi = max(ec.max(), 1e-9)
    a = int(best_action[sel])
    bars = "".join(
        f"<div class='r {'chosen' if m==a else ''}'><span class='lbl'>{ACTIONS[m]}</span>"
        f"<div class='bar'><i style='width:{ec[m]/hi*100:.1f}%'></i></div>"
        f"<span class='val'>{ec[m]:,.1f}</span></div>" for m in range(5))
    st.markdown(f"<div class='dec'>{bars}</div>", unsafe_allow_html=True)

    likely = int(P[sel].argmax())
    if likely != a:
        st.markdown(
            f"<div class='disagree'><strong>The likeliest outcome is not the right "
            f"action.</strong><br>This truck is most likely fine "
            f"({CLASS_WINDOW[likely].lower()}, {P[sel,likely]*100:.1f}%). But a "
            f"{P[sel,4]*100:.1f}% chance of imminent failure costs "
            f"{P[sel,4]*COST[4,0]:,.0f} on average if ignored, while the check "
            f"costs {COST[0,a]:.0f}. Acting is cheaper.</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='agree'>Most likely outcome and cheapest action agree: "
            f"<strong>{ACTIONS[a].lower()}</strong>.</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
st.markdown("<div class='sect'>What drives the prediction</div>",
            unsafe_allow_html=True)
tf = pd.DataFrame(meta["top_features"]).head(8)
tf["share"] = tf["gain"] / tf["gain"].sum()
st.markdown("".join(
    f"<div class='r' style='display:flex;align-items:center;gap:.7rem;margin:.25rem 0'>"
    f"<span class='lbl' style=\"font-family:'IBM Plex Mono',monospace;font-size:.75rem;"
    f"width:14rem;color:var(--muted)\">{r.feature}</span>"
    f"<div style='flex:1;height:13px;background:#DCE0E3'>"
    f"<div style='height:100%;width:{r.share*100:.1f}%;background:#9AA3A8'></div></div></div>"
    for r in tf.itertuples()), unsafe_allow_html=True)

st.markdown(
    "<p class='note' style='margin-top:1.6rem'>"
    "Component age and cumulative usage counters carry most of the signal, which "
    "matches a wear-out failure mode. Sensor names are anonymised in the source "
    "dataset.<br><br>"
    "Data: SCANIA Component X (IDA 2024 Industrial Challenge, CC BY 4.0). "
    "Model probabilities are precomputed; the decision layer above is live.</p>",
    unsafe_allow_html=True)
