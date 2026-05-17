"""Read every overnight/results/**/*.json and write OVERNIGHT_REPORT.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from overnight.eval_common import REPO  # noqa: E402

RESULTS = REPO / "overnight" / "results"
REPORT = REPO / "OVERNIGHT_REPORT.md"


def load_stage1():
    rows = []
    for p in sorted((RESULTS / "stage1").glob("*.json")):
        try:
            d = json.loads(p.read_text())
            rows.append({
                "variant": p.stem,
                "n_train_visible": d.get("_meta", {}).get("epochs"),
                "ood_f1":    d["ood_overall"]["f1_macro"],
                "ood_acc":   d["ood_overall"]["accuracy"],
                "manual_acc": d["manual"]["accuracy"],
                "manual_acc_eq":  d["manual"].get("acc_equivalent"),
                "manual_acc_uneq": d["manual"].get("acc_unequivalent"),
                "claude_acc": d["claude"]["accuracy"],
                "claude_pred_uneq_share": d["claude"]["pred_balance"].get("1", 0) / d["claude"]["n"],
                "elapsed_sec": d.get("_meta", {}).get("elapsed_sec"),
            })
        except Exception as e:
            rows.append({"variant": p.stem, "error": str(e)})
    return rows


def load_stage2():
    out = {}
    for p in sorted((RESULTS / "stage2").glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except Exception as e:
            out[p.stem] = {"error": str(e)}
    return out


def load_stage3():
    p = RESULTS / "stage3" / "timeline.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["timeline"]
    except Exception as e:
        return {"error": str(e)}


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "  -  "
    return f"{x*100:5.1f}%"


def fmt_f1(x: float | None) -> str:
    if x is None:
        return "  -  "
    return f"{x:5.3f}"


def render_stage1(rows: list[dict]) -> str:
    if not rows:
        return "_(no stage 1 results yet)_\n\n"
    rows_sorted = sorted(
        [r for r in rows if "error" not in r],
        key=lambda r: r["manual_acc"], reverse=True,
    )
    lines = [
        "| Variant | OOD F1 | OOD acc | Manual acc | Eq / Uneq | Claude acc | Claude %uneq | sec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_sorted:
        lines.append("| {v} | {f1} | {oa} | {ma} | {eq}/{un} | {ca} | {cs} | {s} |".format(
            v=r["variant"],
            f1=fmt_f1(r["ood_f1"]),
            oa=fmt_pct(r["ood_acc"]),
            ma=fmt_pct(r["manual_acc"]),
            eq=fmt_pct(r.get("manual_acc_eq")),
            un=fmt_pct(r.get("manual_acc_uneq")),
            ca=fmt_pct(r["claude_acc"]),
            cs=fmt_pct(r["claude_pred_uneq_share"]),
            s=int(r["elapsed_sec"]) if r["elapsed_sec"] else "-",
        ))
    err_rows = [r for r in rows if "error" in r]
    if err_rows:
        lines.append("")
        lines.append("**Errors:**")
        for r in err_rows:
            lines.append(f"- `{r['variant']}`: {r['error']}")
    return "\n".join(lines) + "\n\n"


def render_stage2(data: dict) -> str:
    if not data:
        return "_(no stage 2 results yet)_\n\n"
    out = []
    for name, d in data.items():
        if "error" in d:
            out.append(f"### {name}\n\nerror: {d['error']}\n")
            continue
        out.append(f"### {name}")
        out.append(f"_checkpoint:_ `{d['checkpoint']}`")
        out.append(f"_samples:_ synthetic={d['n_synthetic']}, manual={d['n_manual']}, claude={d['n_claude']}")
        out.append("")
        out.append("| Layer | linear val | linear manual | linear claude | mlp val | mlp manual | neg val | ablated val | ablated manual | shortcut tax |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in d["by_layer"]:
            le = r["linear_eq"]; me = r["mlp_eq"]
            ab = r.get("linear_eq_ablated") or {}
            out.append("| {l:>2} | {lv} | {lm} | {lc} | {mv} | {mm} | {nv} | {av} | {am} | {st:+.3f} |".format(
                l=r["layer"],
                lv=fmt_f1(le["val"]), lm=fmt_f1(le["manual"]), lc=fmt_f1(le["claude"]),
                mv=fmt_f1(me["val"]), mm=fmt_f1(me["manual"]),
                nv=fmt_f1(r.get("linear_neg_val")),
                av=fmt_f1(ab.get("val")) if ab else "  -  ",
                am=fmt_f1(ab.get("manual")) if ab else "  -  ",
                st=r.get("shortcut_tax_linear", 0.0),
            ))
        out.append("")
    return "\n".join(out) + "\n"


def render_stage3(timeline) -> str:
    if timeline is None:
        return "_(no stage 3 results yet)_\n\n"
    if isinstance(timeline, dict) and "error" in timeline:
        return f"error: {timeline['error']}\n\n"
    out = [
        "| Step | OOD F1 | Manual acc | Manual eq | Manual uneq | Claude acc | Claude eq | Claude uneq |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in timeline:
        out.append("| {s} | {of} | {ma} | {meq} | {muneq} | {ca} | {ceq} | {cuneq} |".format(
            s=r["step"],
            of=fmt_f1(r["ood_f1"]),
            ma=fmt_pct(r["manual_acc"]),
            meq=fmt_pct(r.get("manual_acc_eq")),
            muneq=fmt_pct(r.get("manual_acc_uneq")),
            ca=fmt_pct(r["claude_acc"]),
            ceq=fmt_pct(r.get("claude_acc_eq")),
            cuneq=fmt_pct(r.get("claude_acc_uneq")),
        ))
    return "\n".join(out) + "\n"


def main() -> int:
    s1 = load_stage1()
    s2 = load_stage2()
    s3 = load_stage3()

    parts = ["# Overnight experiment report\n"]
    parts.append("Generated by `overnight/summarize.py`. Live-updated; rerun any time.\n\n")

    parts.append("## Stage 1: data-mitigation sweep\n\n")
    parts.append("Ranked by manual-set accuracy (the OOD-style holdout). Higher manual_acc with "
                 "OOD F1 not collapsing = the shortcut is partially broken.\n\n")
    parts.append(render_stage1(s1))

    parts.append("## Stage 2: per-layer probes + negation-direction ablation\n\n")
    parts.append("Negation direction is the logreg coefficient against `contains_negation_in_synthetic`. "
                 "Ablated columns show equivalence-probe AUC after projecting that direction out of [CLS]. "
                 "If ablated val AUC barely drops the negation feature was redundant; if manual AUC *rises* "
                 "after ablation the model was hurt by the shortcut.\n\n")
    parts.append(render_stage2(s2))

    parts.append("## Stage 3: training dynamics of the baseline\n\n")
    parts.append("Per-step checkpoint scores. Watch for divergence between OOD F1 (climbs) and "
                 "manual_acc (may peak early then drop).\n\n")
    parts.append(render_stage3(s3))

    REPORT.write_text("".join(parts))
    print(f"[summarize] wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
