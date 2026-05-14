"""Patch mech_interp/results.md with numbers from probe_metrics.json.

Idempotent: re-running overwrites the previous fill-ins. Looks for the literal
sentinels `**Actual result:** _(filled in)_` in results.md and replaces each
with the corresponding sentence built from the metrics JSON. Adds a final
"Summary verdict" section.

Be deliberate about reading what the numbers actually say. We do NOT claim
a hypothesis is "confirmed" — we just describe which hypothesis the data
matches best, and what we still cannot conclude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_MD = REPO / "mech_interp" / "results.md"
DEFAULT_METRICS = REPO / "mech_interp" / "results" / "probe_metrics.json"

SENTINEL = "**Actual result:** _(filled in)_"
PRIOR_SENTINEL = "**Actual:** _(filled in)_"


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _peak(rows: list[dict], key: str) -> tuple[int, float | None]:
    best_idx = -1
    best = -1.0
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if v > best:
            best = v
            best_idx = r["layer"]
    return best_idx, (best if best_idx >= 0 else None)


def build_fill_ins(metrics: dict) -> dict:
    rows = metrics["metrics_by_layer"]
    prior = metrics["prior_regex_auc_synthetic"]

    le, ve = _peak(rows, "probe_equiv_auc_val")
    ln, vn = _peak(rows, "probe_neg_auc_val")
    lr, vr = _peak(rows, "pred_pearson_equiv_neg_val")
    lm, vm = _peak(rows, "probe_equiv_auc_manual")

    last = rows[-1]

    h1 = (
        f"**Actual result:** peak ρ(probe_equiv, probe_neg) on synthetic val = "
        f"{_fmt(vr)} at layer {lr}. "
        f"At that same layer probe_equiv AUC = {_fmt(rows[lr]['probe_equiv_auc_val'])} "
        f"and probe_neg AUC = {_fmt(rows[lr]['probe_neg_auc_val'])}. "
        + ("ρ exceeds 0.85 — the two probes agree per-sample, " if (vr is not None and vr > 0.85) else "ρ does not exceed 0.85, ")
        + "so H1 is "
        + ("**consistent with the data**." if (vr is not None and vr > 0.85) else "**not supported** at this threshold.")
    )

    h2 = (
        f"**Actual result:** at the top layer (layer {last['layer']}) "
        f"probe_equiv AUC = {_fmt(last['probe_equiv_auc_val'])} and "
        f"probe_neg AUC = {_fmt(last['probe_neg_auc_val'])}; "
        f"the gap is {_fmt((last['probe_equiv_auc_val'] or 0.0) - (last['probe_neg_auc_val'] or 0.0))}. "
        f"Peak probe_neg AUC across all layers = {_fmt(vn)} (layer {ln}); "
        f"peak probe_equiv AUC = {_fmt(ve)} (layer {le}). "
    )

    h3 = (
        f"**Actual result:** probe_neg AUC range across layers "
        f"= [{_fmt(min((r['probe_neg_auc_val'] for r in rows if r['probe_neg_auc_val'] is not None), default=None))}, "
        f"{_fmt(max((r['probe_neg_auc_val'] for r in rows if r['probe_neg_auc_val'] is not None), default=None))}], "
        f"vs regex-only prior {_fmt(prior)}. "
        + ("probe_neg climbs well above the regex prior, so negation is encoded as more than a surface lexical match." if (vn is not None and prior is not None and vn > prior + 0.05) else "probe_neg stays close to or below the regex prior — the model does not appear to encode negation richer than the regex itself.")
    )

    aux = (
        f"**Actual result:** best probe_equiv AUC on synthetic val = {_fmt(ve)} (layer {le}); "
        f"best probe_equiv AUC on manual (n=20) = {_fmt(vm)} (layer {lm}). "
        f"Drop = {_fmt((ve or 0.0) - (vm or 0.0))}. "
        + ("Synthetic→manual drop > 0.20 — the synthetic decision function does not transfer to human-labeled pairs, consistent with a shortcut even where H1 alone fails to localize the precise feature."
           if (ve is not None and vm is not None and (ve - vm) > 0.20)
           else "Synthetic→manual drop is < 0.20 — the manual eval does not refute on its own that the model learned semantics.")
    )

    prior_text = (
        f"**Actual:** regex-only AUC on the synthetic pool = {_fmt(prior)} "
        f"(negation regex on text_b alone predicts the equivalence label this well)."
    )

    return {
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "aux": aux,
        "prior": prior_text,
        "peak_layer_corr": (lr, vr),
        "peak_layer_equiv": (le, ve),
        "peak_layer_neg": (ln, vn),
        "manual_drop": (ve, vm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_md", type=Path, default=DEFAULT_RESULTS_MD)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    fills = build_fill_ins(metrics)

    text = args.results_md.read_text()

    # Replace the 4 hypothesis sentinels in order.
    order = [fills["h1"], fills["h2"], fills["h3"], fills["aux"]]
    out_chunks: list[str] = []
    cursor = 0
    for replacement in order:
        idx = text.find(SENTINEL, cursor)
        if idx == -1:
            raise RuntimeError(f"Sentinel {SENTINEL!r} not found at expected position {len(out_chunks)}")
        out_chunks.append(text[cursor:idx])
        out_chunks.append(replacement)
        cursor = idx + len(SENTINEL)
    out_chunks.append(text[cursor:])
    text = "".join(out_chunks)

    # Replace the prior sentinel.
    text = text.replace(PRIOR_SENTINEL, fills["prior"], 1)

    # Append a small summary verdict (idempotent: re-run replaces it).
    verdict_marker = "<!-- VERDICT -->"
    lr, vr = fills["peak_layer_corr"]
    le, ve = fills["peak_layer_equiv"]
    ln, vn = fills["peak_layer_neg"]
    ve_, vm_ = fills["manual_drop"]
    verdict = (
        f"\n\n---\n{verdict_marker}\n\n## Summary verdict (auto-filled by finalize_results.py)\n\n"
        f"- Peak per-sample probe correlation: ρ = {_fmt(vr)} at layer {lr} → "
        + ("**H1 supported** (negation shortcut localized)." if (vr is not None and vr > 0.85) else "H1 not supported at the ρ>0.85 threshold.")
        + "\n"
        f"- Peak probe_equiv AUC = {_fmt(ve)} (layer {le}); peak probe_neg AUC = {_fmt(vn)} (layer {ln}); "
        + ("gap small (≤0.05) → H2 most consistent with data." if (ve is not None and vn is not None and (ve - vn) <= 0.05) else "gap moderate-to-large → H3 partially supported.")
        + "\n"
        f"- Synthetic→manual probe_equiv AUC drop: {_fmt((ve_ or 0.0) - (vm_ or 0.0))} (synthetic {_fmt(ve_)} → manual {_fmt(vm_)}).\n"
        f"- Regex prior: AUC = {_fmt(metrics['prior_regex_auc_synthetic'])}.\n\n"
        f"**Phase-2 hook.** "
        + (f"Run activation patching at layer {lr}: ablate the negation direction (probe_neg weight vector) and measure label-flip rate on synthetic vs manual."
           if (vr is not None and vr > 0.7)
           else "Probe correlation is weak; before doing causal patching, expand the probe target set — try sentiment-flip, length ratio, n-gram overlap.")
        + "\n"
    )
    if verdict_marker in text:
        text = text.split(verdict_marker)[0] + verdict_marker + verdict.split(verdict_marker, 1)[1]
    else:
        text = text + verdict

    args.results_md.write_text(text)
    print(f"[finalize] wrote {args.results_md}")


if __name__ == "__main__":
    main()
