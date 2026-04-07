from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
ACTIVITY_FALLBACK_LABEL = "Non annoté"
FAMILY_COLORS = {
    "judo > randoris > TW": "#e63946",
    "judo > randoris > NW pure": "#457b9d",
    "judo > randoris > libres": "#e76f51",
    "judo > technique": "#2a9d8f",
    "prépa": "#6c63ff",
}
CORR_COLORS = {
    "division": "#8338ec",
    "fusion": "#f4a261",
    "interpolation_pchip": "#e63946",
    "interpolation_lineaire": "#457b9d",
}
ARTEFACT_LABELS = ["manque", "faux_battement", "long", "court", "artefact_absolu"]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def rel_path(root: Path, value, fallback: Path) -> Path:
    if not value:
        return fallback
    return root / Path(str(value).replace("\\", "/"))


def file_sig(path: Path) -> tuple[str, int, int]:
    if not path.exists():
        return (str(path), -1, -1)
    stat = path.stat()
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


def build_manifest(root: Path = DATA_ROOT) -> tuple[tuple[str, int, int], ...]:
    items = []
    for meta_path in sorted((root / "processed").glob("*/session_meta.json")):
        items.append(file_sig(meta_path))
        try:
            meta = read_json(meta_path)
        except Exception:
            continue
        session_id = str(meta.get("session_id") or meta_path.parent.name)
        rr_path = rel_path(root, meta.get("rr_clean_filepath"), root / "clean" / session_id / "rr_clean.parquet")
        items.append(file_sig(rr_path))
        items.append(file_sig(rr_path.parent / "clean_meta.json"))
    return tuple(items)


def norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "RR_clean" not in out.columns:
        out["RR_clean"] = out["display_rr_ms"] if "display_rr_ms" in out.columns else out.get("rr_interval_ms", np.nan)
    if "offset_s" not in out.columns:
        if "t_offset_clean_ms" in out.columns:
            out["offset_s"] = out["t_offset_clean_ms"].astype("float64") / 1000.0
        else:
            out["offset_s"] = out.get("t_offset_ms", pd.Series(np.arange(len(out))))
            out["offset_s"] = pd.to_numeric(out["offset_s"], errors="coerce") / 1000.0
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.NaT
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    defaults = {
        "label": "ok",
        "run_flag": "ok",
        "run_series_flag": "ok",
        "deco_flag": "ok",
        "correction_flag": "ok",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default).astype(str)
    if out["run_series_flag"].eq("ok").all() and out["run_flag"].eq("run_serie").any():
        out.loc[out["run_flag"].eq("run_serie"), "run_series_flag"] = "run_serie"
    for col in ["fc_ok", "hrr_ok", "rmssd_ok"]:
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].fillna(False).astype(bool)
    out["RR_clean"] = pd.to_numeric(out["RR_clean"], errors="coerce")
    out["offset_s"] = pd.to_numeric(out["offset_s"], errors="coerce")
    out["_id"] = np.arange(len(out))
    return out


def count_groups(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    values = mask.fillna(False).astype(int)
    starts = values.diff().fillna(values.iloc[0]).eq(1)
    if bool(values.iloc[0]):
        starts.iloc[0] = True
    return int(starts.sum())


def dense_ranges(mask: pd.Series, offsets: pd.Series) -> list[tuple[float, float]]:
    flags = mask.fillna(False).to_numpy(dtype=bool)
    xs = pd.to_numeric(offsets, errors="coerce").to_numpy(dtype="float64")
    diffs = np.diff(xs[np.isfinite(xs)])
    diffs = diffs[diffs > 0]
    pad = float(np.median(diffs)) if len(diffs) else 0.5
    start = None
    out = []
    for i, active in enumerate(flags):
        if active and start is None:
            start = i
        at_end = i == len(flags) - 1 or not flags[i + 1]
        if active and start is not None and at_end and np.isfinite(xs[start]) and np.isfinite(xs[i]):
            out.append((float(xs[start] - pad / 2.0), float(xs[i] + pad / 2.0)))
            start = None
    return out


def compute_quality_metrics(df: pd.DataFrame, meta: dict) -> dict:
    df = norm(df)
    duration = float(meta.get("duree_s") or 0.0)
    gaps = meta.get("gap_durations_s") or []
    deco_total_s = float(np.nansum(gaps)) if gaps else 0.0
    analysable = int(df["label"].ne("gap_deco").sum())
    art_mask = df["label"].ne("ok") & df["label"].ne("gap_deco")
    labels = {lab: ((df["label"].eq(lab).sum() / analysable) * 100.0 if analysable else 0.0) for lab in ARTEFACT_LABELS}
    return {
        "nb_deco": int(meta.get("gaps_count") or 0),
        "deco_total_s": deco_total_s,
        "deco_pct": (deco_total_s / duration) * 100.0 if duration > 0 else 0.0,
        "pct_artefacts": (art_mask.sum() / analysable) * 100.0 if analysable else 0.0,
        "labels_detail": labels,
        "nb_zones_denses": count_groups(df["run_series_flag"].eq("run_serie")),
        "nb_cassures_artefact": count_groups(df["run_flag"].eq("gap_artefact")),
        "nb_cassures_deco": count_groups(df["deco_flag"].eq("gap_deco_long")),
        "pct_fc_ok": (df["fc_ok"].sum() / len(df)) * 100.0 if len(df) else 0.0,
        "pct_hrr_ok": (df["hrr_ok"].sum() / len(df)) * 100.0 if len(df) else 0.0,
        "pct_rmssd_ok": (df["rmssd_ok"].sum() / len(df)) * 100.0 if len(df) else 0.0,
        "corrections": {name: int(df["correction_flag"].eq(name).sum()) for name in CORR_COLORS},
    }


@st.cache_data(show_spinner=False)
def load_all_sessions(manifest: tuple[tuple[str, int, int], ...], root_str: str = str(DATA_ROOT)) -> list[dict]:
    del manifest
    root = Path(root_str)
    rows = []
    for meta_path in sorted((root / "processed").glob("*/session_meta.json")):
        try:
            meta = read_json(meta_path)
        except Exception:
            continue
        if bool(meta.get("is_archived")):
            continue
        session_id = str(meta.get("session_id") or meta_path.parent.name)
        rr_path = rel_path(root, meta.get("rr_clean_filepath"), root / "clean" / session_id / "rr_clean.parquet")
        clean_meta_path = rr_path.parent / "clean_meta.json"
        row = {
            "session_id": session_id,
            "date": str(meta.get("date") or ""),
            "heure_debut": str(meta.get("heure_debut") or ""),
            "duree_s": float(meta.get("duree_s") or 0.0),
            "nb_battements_rr": int(meta.get("nb_battements_rr") or 0),
            "activity_label": str(meta.get("activity_label") or "").strip() or ACTIVITY_FALLBACK_LABEL,
            "activity_family": meta.get("activity_family"),
            "judo_session_type": meta.get("judo_session_type"),
            "rr_clean_filepath": str(rr_path),
            "has_clean": False,
            "clean_missing_reason": None,
            "algo_version": None,
            "metrics": None,
        }
        if not rr_path.exists():
            row["clean_missing_reason"] = "clean manquant"
            rows.append(row)
            continue
        try:
            rr_df = pd.read_parquet(rr_path)
            clean_meta = read_json(clean_meta_path) if clean_meta_path.exists() else {}
            row["algo_version"] = clean_meta.get("algo_version") or clean_meta.get("cleaning_algo_version")
            row["metrics"] = compute_quality_metrics(rr_df, meta)
            row["has_clean"] = True
        except Exception:
            row["clean_missing_reason"] = "clean illisible"
        rows.append(row)
    return sorted(rows, key=lambda x: (x["date"], x["heure_debut"], x["session_id"]))


@st.cache_data(show_spinner=False)
def load_rr_frame(path_str: str, signature: tuple[str, int, int]) -> pd.DataFrame:
    del signature
    return pd.read_parquet(path_str)


def qual_color(pct: float, thresholds: tuple[float, float] = (90.0, 70.0)) -> str:
    if pct >= thresholds[0]:
        return "#22c55e"
    if pct >= thresholds[1]:
        return "#f59e0b"
    return "#ef4444"


def art_color(pct: float) -> str:
    return qual_color(100.0 - pct, (95.0, 85.0))


def fmt_duration(seconds) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "--:--"
    total = int(round(float(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def fmt_short(seconds) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "--"
    total = int(round(float(seconds)))
    return f"{total // 60}m{total % 60:02d}s"


def fmt_axis_mmss(seconds: float) -> str:
    total = int(round(float(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    arr = np.asarray(vals, dtype="float64")
    return float(arr.mean()), float(arr.std(ddof=0))

def render_header(sessions: list[dict], clean_sessions: list[dict]) -> None:
    version = next((str(s["algo_version"]) for s in reversed(clean_sessions) if s.get("algo_version")), "N/A")
    st.markdown(
        f"**Projet I — INSEP · Avril 2026**\n\n# Bilan qualité du signal RR\n{len(sessions)} séances collectées · Pipeline v{version} · Polar H10 + PSL"
    )


def render_badges(sessions: list[dict]) -> None:
    counts = Counter(s["activity_label"] for s in sessions)
    html = []
    for label, count in sorted(counts.items(), key=lambda item: item[0].casefold()):
        color = FAMILY_COLORS.get(label, "#64748b")
        html.append(
            f'<span style="display:inline-block;padding:0.35rem 0.7rem;margin:0.15rem;border-radius:999px;background:{color};color:white;">{label} · {count}</span>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def render_kpis(sessions: list[dict]) -> None:
    if not sessions:
        st.info("Aucune séance clean sur ce filtre.")
        return
    dur_m, dur_s = mean_std([s["duree_s"] for s in sessions])
    deco_s, _ = mean_std([s["metrics"]["deco_total_s"] for s in sessions])
    deco_pct, _ = mean_std([s["metrics"]["deco_pct"] for s in sessions])
    nb_m, nb_s = mean_std([s["metrics"]["nb_deco"] for s in sessions])
    art_m, art_s = mean_std([s["metrics"]["pct_artefacts"] for s in sessions])
    dense_m, dense_s = mean_std([s["metrics"]["nb_zones_denses"] for s in sessions])
    cass_m, _ = mean_std([s["metrics"]["nb_cassures_artefact"] for s in sessions])
    fc_m, fc_s = mean_std([s["metrics"]["pct_fc_ok"] for s in sessions])
    rmssd_m, rmssd_s = mean_std([s["metrics"]["pct_rmssd_ok"] for s in sessions])
    cards = [
        ("Durée moy.", fmt_duration(dur_m), "", f"± {fmt_duration(dur_s)}", "#ffffff", "rgba(30,41,59,0.84)", "rgba(148,163,184,0.32)"),
        ("Déco. moy.", f"{deco_s / 60.0:.1f}", "min", f"{deco_pct:.1f}% de la séance", "#f59e0b", "rgba(120,53,15,0.84)", "rgba(245,158,11,0.35)"),
        ("Nb déco. moy.", f"{nb_m:.1f}", "", f"± {nb_s:.1f}", "#ffffff", "rgba(51,65,85,0.84)", "rgba(148,163,184,0.32)"),
        ("% artefacts RR", f"{art_m:.1f}", "%", f"± {art_s:.1f}%", art_color(art_m), "rgba(22,101,52,0.82)" if art_m < 5 else "rgba(120,53,15,0.82)", "rgba(34,197,94,0.28)" if art_m < 5 else "rgba(245,158,11,0.32)"),
        ("Zones denses", f"{dense_m:.1f}", "", f"± {dense_s:.1f}", "#ff5864", "rgba(127,29,29,0.84)", "rgba(239,68,68,0.34)"),
        ("Cassures art.", f"{cass_m:.1f}", "", "run > 15 consec.", "#ff5864", "rgba(127,29,29,0.84)", "rgba(239,68,68,0.34)"),
        ("FC exploitable", f"{fc_m:.1f}", "%", f"± {fc_s:.1f}%", qual_color(fc_m), "rgba(20,83,45,0.84)", "rgba(34,197,94,0.32)"),
        ("RMSSD exploit.", f"{rmssd_m:.1f}", "%", f"± {rmssd_s:.1f}%", qual_color(rmssd_m), "rgba(20,83,45,0.84)", "rgba(34,197,94,0.32)"),
    ]
    html = []
    for label, value, unit, sub, color, bg_color, border_color in cards:
        html.append(
            f'<div style="flex:1 1 180px;min-width:180px;padding:0.95rem;border-radius:16px;background:{bg_color};border:1px solid {border_color};box-shadow:0 10px 24px rgba(15,23,42,0.16);">'
            f'<div style="font-size:0.76rem;color:#e2e8f0;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">{label}</div>'
            f'<div style="font-family:Consolas,monospace;font-size:1.55rem;font-weight:800;color:{color};text-shadow:0 1px 0 rgba(15,23,42,0.22);">{value} <span style="font-size:1rem;color:#f8fafc;">{unit}</span></div>'
            f'<div style="font-size:0.82rem;color:#f8fafc;font-weight:600;">{sub}</div></div>'
        )
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:0.7rem;">{"".join(html)}</div>', unsafe_allow_html=True)


def session_filter(sessions: list[dict], choice: str) -> list[dict]:
    clean = [s for s in sessions if s["has_clean"]]
    if choice == "Randoris":
        return [s for s in clean if s.get("judo_session_type") == "randoris"]
    if choice == "Technique":
        return [s for s in clean if s.get("judo_session_type") == "technique"]
    if choice == "Prépa":
        return [s for s in clean if s.get("activity_family") == "prepa"]
    return clean


def quality_bucket(pct: float, thresholds: tuple[float, float] = (90.0, 70.0)) -> str:
    if pct >= thresholds[0]:
        return "good"
    if pct >= thresholds[1]:
        return "warn"
    return "bad"


def chart_data(sessions: list[dict], key: str, inverse: bool = False) -> pd.DataFrame:
    counts = Counter(s["date"] for s in sessions)
    rows = []
    for i, s in enumerate(sessions):
        mmdd = s["date"][5:] if len(s["date"]) >= 10 else s["date"]
        label = f"{mmdd} {s['heure_debut'][:5]}" if counts[s["date"]] > 1 else mmdd
        value = float(s["metrics"][key])
        quality = quality_bucket(100.0 - value, (95.0, 85.0)) if inverse else quality_bucket(value, (90.0, 70.0))
        rows.append({"label": label, "order": i, "value": value, "quality": quality})
    return pd.DataFrame(rows)


def chart_domain(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return 0.0, 100.0
    vmin = float(values.min())
    vmax = float(values.max())
    low = max(0.0, float(np.floor(vmin - 5.0)))
    high = min(100.0, float(np.ceil(vmax + 5.0)))
    if high <= low:
        high = min(100.0, low + 10.0)
    return low, high


def render_chart_legend(mean_value: float) -> None:
    st.markdown(
        (
            '<div style="display:flex;flex-wrap:wrap;gap:0.55rem;align-items:center;margin:0.05rem 0 0.45rem 0;">'
            '<span style="display:inline-flex;align-items:center;gap:0.35rem;padding:0.22rem 0.55rem;border-radius:999px;background:rgba(34,197,94,0.14);color:#f8fafc;font-weight:700;font-size:0.82rem;">'
            '<span style="width:0.7rem;height:0.7rem;border-radius:999px;background:#22c55e;display:inline-block;"></span>Bon</span>'
            '<span style="display:inline-flex;align-items:center;gap:0.35rem;padding:0.22rem 0.55rem;border-radius:999px;background:rgba(245,158,11,0.16);color:#f8fafc;font-weight:700;font-size:0.82rem;">'
            '<span style="width:0.7rem;height:0.7rem;border-radius:999px;background:#f59e0b;display:inline-block;"></span>Intermédiaire</span>'
            '<span style="display:inline-flex;align-items:center;gap:0.35rem;padding:0.22rem 0.55rem;border-radius:999px;background:rgba(239,68,68,0.16);color:#f8fafc;font-weight:700;font-size:0.82rem;">'
            '<span style="width:0.7rem;height:0.7rem;border-radius:999px;background:#ef4444;display:inline-block;"></span>À surveiller</span>'
            f'<span style="display:inline-flex;align-items:center;gap:0.42rem;padding:0.25rem 0.7rem;border-radius:999px;background:rgba(226,232,240,0.18);border:1px solid rgba(226,232,240,0.32);color:#ffffff;font-weight:800;font-size:0.84rem;">'
            f'<span style="width:1rem;height:0.16rem;border-radius:999px;background:#ffffff;display:inline-block;"></span>Moyenne : {mean_value:.1f}%</span>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_chart(sessions: list[dict], key: str, title: str, inverse: bool = False) -> None:
    df = chart_data(sessions, key, inverse)
    if df.empty:
        st.info("Aucune donnée à afficher.")
        return
    quality_scale = alt.Scale(domain=["good", "warn", "bad"], range=["#22c55e", "#f59e0b", "#ef4444"])
    y_min, y_max = chart_domain(df["value"])
    mean_value = float(df["value"].mean())
    x_sort = df["label"].tolist()
    render_chart_legend(mean_value)
    base = alt.Chart(df).encode(
        x=alt.X("label:N", sort=x_sort, axis=alt.Axis(labelAngle=-20, labelColor="#94a3b8", titleColor="#64748b"), title="Séance"),
        tooltip=[alt.Tooltip("label:N", title="Séance"), alt.Tooltip("value:Q", title="Valeur", format=".2f")],
    )
    bars = base.mark_bar(size=42, cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.96).encode(
        y=alt.Y("value:Q", title="%", scale=alt.Scale(domain=[y_min, y_max]), axis=alt.Axis(labelColor="#94a3b8", titleColor="#64748b", gridColor="#e2e8f0")),
        y2=alt.Y2Datum(y_min),
        color=alt.Color("quality:N", scale=quality_scale, legend=None)
    )
    labels = base.mark_text(dy=-8, color="#cbd5e1", fontWeight="bold").encode(
        y=alt.Y("value:Q", scale=alt.Scale(domain=[y_min, y_max])),
        text=alt.Text("value:Q", format=".1f")
    )
    mean_df = pd.DataFrame({"mean": [mean_value], "label": [f"Moy. {mean_value:.1f}%"]})
    mean_rule = alt.Chart(mean_df).mark_rule(color="#ffffff", strokeDash=[6, 4], strokeWidth=1.6, opacity=0.95).encode(y="mean:Q")
    mean_text = alt.Chart(mean_df).mark_text(align="left", dx=8, dy=-8, color="#111827", fontWeight="bold", fontSize=12).encode(x=alt.value(8), y="mean:Q", text="label:N")
    chart = (bars + labels + mean_rule + mean_text).properties(height=265, title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

def summary_df(sessions: list[dict]) -> pd.DataFrame:
    rows = []
    for s in sessions:
        m = s["metrics"]
        rows.append({
            "session_id": s["session_id"],
            "Date": s["date"],
            "Type": s["activity_label"],
            "Durée": fmt_duration(s["duree_s"]),
            "Déco.": f"{m['nb_deco']} ({fmt_short(m['deco_total_s'])})",
            "% Art.": m["pct_artefacts"],
            "Zones D.": m["nb_zones_denses"],
            "Cass.": m["nb_cassures_artefact"] + m["nb_cassures_deco"],
            "FC ok": m["pct_fc_ok"],
            "RMSSD ok": m["pct_rmssd_ok"],
        })
    return pd.DataFrame(rows)


def export_df(sessions: list[dict]) -> pd.DataFrame:
    rows = []
    for s in sessions:
        m = s.get("metrics") or {}
        row = {k: s.get(k) for k in ["session_id", "date", "heure_debut", "activity_label", "activity_family", "judo_session_type", "duree_s", "nb_battements_rr", "algo_version", "rr_clean_filepath"]}
        row["clean_status"] = "ok" if s["has_clean"] else s["clean_missing_reason"]
        row.update({
            "nb_deco": m.get("nb_deco"), "deco_total_s": m.get("deco_total_s"), "deco_pct": m.get("deco_pct"),
            "pct_artefacts": m.get("pct_artefacts"), "nb_zones_denses": m.get("nb_zones_denses"),
            "nb_cassures_artefact": m.get("nb_cassures_artefact"), "nb_cassures_deco": m.get("nb_cassures_deco"),
            "pct_fc_ok": m.get("pct_fc_ok"), "pct_hrr_ok": m.get("pct_hrr_ok"), "pct_rmssd_ok": m.get("pct_rmssd_ok"),
        })
        for lab in ARTEFACT_LABELS:
            row[f"pct_{lab}"] = m.get("labels_detail", {}).get(lab)
        for name in CORR_COLORS:
            row[f"count_{name}"] = m.get("corrections", {}).get(name)
        rows.append(row)
    return pd.DataFrame(rows)


def style_summary(df: pd.DataFrame):
    if df.empty:
        return df.style
    styler = df.drop(columns=["session_id"], errors="ignore").style.format({"% Art.": "{:.1f}%", "FC ok": "{:.1f}%", "RMSSD ok": "{:.1f}%"})
    styler = styler.map(lambda v: f"background-color:{FAMILY_COLORS.get(v, '#64748b')}22;border-left:0.35rem solid {FAMILY_COLORS.get(v, '#64748b')};", subset=["Type"])
    styler = styler.map(lambda v: f"color:{art_color(v)};font-weight:700;", subset=["% Art."])
    styler = styler.map(lambda v: f"color:{qual_color(v)};font-weight:700;", subset=["FC ok", "RMSSD ok"])
    return styler

def plot_rr_clean(session: dict, compact: bool = False) -> go.Figure:
    frame = norm(load_rr_frame(session["rr_clean_filepath"], file_sig(Path(session["rr_clean_filepath"]))))
    line = frame.loc[frame["correction_flag"].ne("not_cleaned") & frame["RR_clean"].notna() & frame["offset_s"].notna()].copy()
    corrected = line.loc[~line["correction_flag"].isin(["ok", "not_cleaned"])].copy()
    if len(line) > 15000:
        step = max(int(np.ceil(len(line) / 2000.0)), 1)
        line = pd.concat([line.iloc[::step], corrected]).sort_values("offset_s").drop_duplicates(subset=["_id"])

    fig = go.Figure()
    for y in [400, 600, 800, 1000]:
        fig.add_hline(y=y, line_color="rgba(100,116,139,0.35)", line_width=1)
    for x0, x1 in dense_ranges(frame["run_series_flag"].eq("run_serie"), frame["offset_s"]):
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(230,57,70,0.12)", line_color="rgba(230,57,70,0.35)", line_width=1)

    if not line.empty:
        fig.add_trace(
            go.Scatter(
                x=line["offset_s"],
                y=line["RR_clean"],
                mode="lines",
                name="RR clean",
                line={"color": "#8ecae6", "width": 1.25},
                opacity=0.7,
                hovertemplate="Offset %{x:.1f}s<br>RR %{y:.0f} ms<extra></extra>",
            )
        )

    for name, color in CORR_COLORS.items():
        part = corrected.loc[corrected["correction_flag"].eq(name)]
        if not part.empty:
            fig.add_trace(
                go.Scatter(
                    x=part["offset_s"],
                    y=part["RR_clean"],
                    mode="markers",
                    name=name,
                    marker={"color": color, "size": 5 if compact else 7, "opacity": 0.95},
                    selected={"marker": {"size": 9 if compact else 11, "opacity": 1.0}},
                    unselected={"marker": {"opacity": 0.35}},
                    hovertemplate=f"{name}<br>Offset %{{x:.1f}}s<br>RR %{{y:.0f}} ms<extra></extra>",
                )
            )

    xmax = float(np.nanmax(frame["offset_s"])) if not frame["offset_s"].dropna().empty else 1.0
    tickvals = np.linspace(0.0, max(xmax, 1.0), 6)
    ticktext = [fmt_axis_mmss(x) for x in tickvals]
    fig.update_xaxes(title="offset", range=[0, max(xmax, 1.0)], tickmode="array", tickvals=tickvals, ticktext=ticktext)
    fig.update_yaxes(title="RR (ms)", range=[200, 1200])
    fig.update_layout(
        height=230 if compact else 320,
        margin={"l": 38, "r": 28, "t": 54, "b": 36},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        dragmode="select",
        hovermode="closest",
        title={
            "text": f"{session['date']} — {session['activity_label']}<span style='font-size:0.8em;color:#cbd5e1'> &nbsp;&nbsp; {fmt_duration(session['duree_s'])} · {session['metrics']['pct_artefacts']:.1f}% art.</span>",
            "x": 0.01,
            "xanchor": "left",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    if line.empty:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="Aucun RR clean exploitable", showarrow=False, font={"color": "#94a3b8"})
    return fig


def main() -> None:
    st.set_page_config(page_title="Bilan Qualité RR", layout="wide")
    sessions = load_all_sessions(build_manifest(), str(DATA_ROOT))
    if not sessions:
        st.warning("Aucune séance non archivée trouvée.")
        return

    clean_sessions = [s for s in sessions if s["has_clean"]]
    missing_sessions = [s for s in sessions if not s["has_clean"]]
    render_header(sessions, clean_sessions)
    render_badges(sessions)

    filt = st.radio("Filtre", ["Toutes", "Randoris", "Technique", "Prépa"], horizontal=True)
    filtered = session_filter(sessions, filt)
    if filt == "Randoris":
        missing = [s for s in missing_sessions if s.get("judo_session_type") == "randoris"]
    elif filt == "Technique":
        missing = [s for s in missing_sessions if s.get("judo_session_type") == "technique"]
    elif filt == "Prépa":
        missing = [s for s in missing_sessions if s.get("activity_family") == "prepa"]
    else:
        missing = missing_sessions
    if missing:
        preview = ", ".join(f"{s['date']} ({s['clean_missing_reason']})" for s in missing[:4])
        more = "" if len(missing) <= 4 else f" +{len(missing) - 4} autre(s)"
        st.warning(f"{len(missing)} séance(s) sans export clean exploitable : {preview}{more}. Elles sont exclues des stats.")

    st.caption("KPI")
    render_kpis(filtered)

    csv_df = export_df(sessions)
    if not filtered:
        st.download_button("Télécharger le tableau complet (CSV)", csv_df.to_csv(index=False).encode("utf-8-sig"), "bilan_qualite_rr_sessions.csv", "text/csv")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        render_chart(filtered, "pct_artefacts", "% artefacts par séance", True)
    with c2:
        render_chart(filtered, "pct_fc_ok", "% FC exploitable")
    with c3:
        render_chart(filtered, "pct_rmssd_ok", "% RMSSD exploitable")

    st.caption("Tableau récapitulatif")
    st.dataframe(style_summary(summary_df(filtered)), use_container_width=True, hide_index=True)
    st.download_button("Télécharger le tableau complet (CSV)", csv_df.to_csv(index=False).encode("utf-8-sig"), "bilan_qualite_rr_sessions.csv", "text/csv")

    ids = [s["session_id"] for s in filtered]
    if "bilan_quality_selected_ids" not in st.session_state:
        st.session_state["bilan_quality_selected_ids"] = ids[-1:] if ids else []
    st.session_state["bilan_quality_selected_ids"] = [sid for sid in st.session_state["bilan_quality_selected_ids"] if sid in ids]
    b1, b2 = st.columns(2)
    if b1.button("Tout sélectionner"):
        st.session_state["bilan_quality_selected_ids"] = ids
    if b2.button("Tout désélectionner"):
        st.session_state["bilan_quality_selected_ids"] = []

    labels = {s["session_id"]: f"{s['date']} · {s['heure_debut'][:5]} · {s['activity_label']}" for s in filtered}
    st.multiselect("Séances à tracer", ids, format_func=lambda sid: labels[sid], key="bilan_quality_selected_ids")
    selected = [s for s in filtered if s["session_id"] in st.session_state["bilan_quality_selected_ids"]]

    st.caption("Légende : RR clean · division · fusion · interpolation PCHIP · interpolation linéaire · zone dense. Les graphes RR sont zoomables, déplaçables et les points corrigés sont sélectionnables via la barre d'outils Plotly.")
    if not selected:
        st.info("Sélectionnez une ou plusieurs séances pour afficher les graphes RR")
        return

    config = {"displaylogo": False, "scrollZoom": True, "modeBarButtonsToAdd": ["select2d", "lasso2d"]}
    if len(selected) == 1:
        st.plotly_chart(plot_rr_clean(selected[0], False), use_container_width=True, config=config)
        return

    cols = st.columns(2)
    for i, session in enumerate(selected):
        with cols[i % 2]:
            st.plotly_chart(plot_rr_clean(session, True), use_container_width=True, config=config)


if __name__ == "__main__":
    main()








