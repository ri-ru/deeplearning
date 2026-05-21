"""Generate poster figures from overnight JSON results.

Writes 6 PNGs to poster_figures/:
  1_eval_matrix.png              cross-train x cross-eval heatmap
  2_per_class_balance.png        eq/uneq accuracy on handwritten n=20
  3_loo_ablation.png             leave-one-out accuracy across 4 eval sets
  4_layer_probe_auc.png          per-layer manual AUC, baseline vs multi-gen
  5_direction_ablation.png       baseline manual AUC vs K directions removed
  6_headline_efficiency.png      52k single-gen vs 800 multi-gen, side-by-side
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "poster_figures"
OUT.mkdir(exist_ok=True)

STAGE14   = json.loads((ROOT / "overnight/results/stage14/matrix.json").read_text())
STAGE14B  = json.loads((ROOT / "overnight/results/stage14b/matrix.json").read_text())
STAGE2_B  = json.loads((ROOT / "overnight/results/stage2/baseline.json").read_text())
STAGE2_M  = json.loads((ROOT / "overnight/results/stage2/best_mitigation.json").read_text())
STAGE6_B  = json.loads((ROOT / "overnight/results/stage6/baseline_layer12.json").read_text())
STAGE3    = json.loads((ROOT / "overnight/results/stage3/timeline.json").read_text())
STAGE5    = json.loads((ROOT / "overnight/results/stage5/fine_timeline.json").read_text())
STAGE11   = json.loads((ROOT / "overnight/results/stage11/ceiling.json").read_text())
STAGE13   = json.loads((ROOT / "overnight/results/stage13/sentence_bert.json").read_text())
STAGE4_DR = json.loads((ROOT / "overnight/results/stage4/davidson_roberta.json").read_text())
STAGE4_DT = json.loads((ROOT / "overnight/results/stage4/davidson_tfidf.json").read_text())
STAGE4_HR = json.loads((ROOT / "overnight/results/stage4/hatexplain_roberta.json").read_text())
STAGE4_HT = json.loads((ROOT / "overnight/results/stage4/hatexplain_tfidf.json").read_text())
PROBE_M   = json.loads((ROOT / "mech_interp/results/probe_metrics.json").read_text())
STAGE1    = {p.stem: json.loads(p.read_text())
             for p in sorted((ROOT / "overnight/results/stage1").glob("v*.json"))}

# Clean academic style
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor":"white",
    "savefig.dpi":      300,
    "font.family":      ["Arial", "DejaVu Sans"],
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "axes.edgecolor":   "#2a2a2a",
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.color":       "#e5e7eb",
    "grid.linewidth":   0.5,
    "xtick.color":      "#2a2a2a",
    "ytick.color":      "#2a2a2a",
    "xtick.direction":  "out",
    "ytick.direction":  "out",
})

INK   = "#1a1a1a"
SLATE = "#6b7a8a"
PALE  = "#c5cdd6"
CORAL = "#d2543d"
TEAL  = "#2e7a78"
NAVY  = "#2e5a87"

RECIPE_LABEL = {
    "poster_baseline":   "baseline (Qwen-72B, 52k)",
    "anthropic_haiku":   "Anthropic Haiku only",
    "openai_gpt4omini":  "GPT-4o-mini only",
    "qwen25_72b":        "Qwen 2.5 72B only",
    "meta_llama3_8b":    "Llama 3.1 8B only",
    "deepseek_v3":       "DeepSeek V3 only",
    "loo_drop_anthropic":"drop Anthropic",
    "loo_drop_openai":   "drop OpenAI",
    "loo_drop_qwen":     "drop Qwen",
    "loo_drop_llama":    "drop Llama",
    "loo_drop_deepseek": "drop DeepSeek",
    "multi_gen":         "multi-gen (all 5)",
}

EVAL_COLS = [
    ("manual_handwritten_n20", "Handwritten\nn=20",   STAGE14B),
    ("claude46_n20",           "Claude 4.6\nn=20",    STAGE14B),
    ("stage7_gpt_n50",         "GPT-4o-mini\nn=50",   STAGE14),
    ("stage10_haiku_n100",     "Claude Haiku\nn=100", STAGE14),
]

ROW_ORDER = [
    "poster_baseline",
    "anthropic_haiku", "openai_gpt4omini", "qwen25_72b", "meta_llama3_8b", "deepseek_v3",
    "loo_drop_anthropic", "loo_drop_openai", "loo_drop_qwen", "loo_drop_llama", "loo_drop_deepseek",
    "multi_gen",
]


def acc(recipe: str, eval_key: str, source: dict) -> float:
    return source[recipe][eval_key]["acc"]


# ----------------------------------------------------------------------------- 1
def fig_eval_matrix():
    rows = ROW_ORDER
    cols = [c[0] for c in EVAL_COLS]
    M = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, (k, _, src) in enumerate(EVAL_COLS):
            M[i, j] = acc(r, k, src)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "slatecoral", ["#f4f6f8", "#cfd6df", "#7d8a99", CORAL]
    )
    norm = mpl.colors.Normalize(vmin=0.40, vmax=1.0)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in EVAL_COLS], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([RECIPE_LABEL[r] for r in rows], fontsize=10)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Evaluation set (human-style writing)")
    ax.set_title("Training recipe x evaluation set: accuracy")

    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            color = "white" if v >= 0.78 else INK
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    # group separators
    for y in [0.5, 5.5, 10.5]:
        ax.axhline(y, color="white", linewidth=2)

    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("accuracy", rotation=270, labelpad=14)
    cbar.outline.set_visible(False)

    fig.text(0.02, 0.06,
             "Baseline + single-gens collapse on human-style evals.\n"
             "multi-gen and drop-DeepSeek are the only recipes that hold.",
             fontsize=9, color="#444")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "1_eval_matrix.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 2
def fig_per_class_balance():
    rows = ["poster_baseline", "anthropic_haiku", "qwen25_72b",
            "meta_llama3_8b", "deepseek_v3", "openai_gpt4omini", "multi_gen"]

    eq    = [STAGE14B[r]["manual_handwritten_n20"]["acc_eq"]   * 100 for r in rows]
    uneq  = [STAGE14B[r]["manual_handwritten_n20"]["acc_uneq"] * 100 for r in rows]
    labels = [RECIPE_LABEL[r] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(rows))
    w = 0.38

    bars_eq   = ax.bar(x - w/2, eq,   w, color=NAVY,  label="equivalent class")
    bars_uneq = ax.bar(x + w/2, uneq, w, color=CORAL, label="unequivalent class")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("per-class accuracy (%)")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Per-class accuracy on handwritten test set (n=20)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, fontsize=10)

    for b, v in list(zip(bars_eq, eq)) + list(zip(bars_uneq, uneq)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"{v:.0f}", ha="center", va="bottom",
                fontsize=9, color=INK)

    fig.text(0.02, -0.04,
             "Single-generator training collapses to one class. "
             "multi-gen (last bar pair) is the only balanced classifier.",
             fontsize=9, color="#444")

    fig.tight_layout()
    fig.savefig(OUT / "2_per_class_balance.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 3
def fig_loo_ablation():
    # Show all 4 eval sets x LOO recipes; multi_gen reference line
    recipes = ["multi_gen", "loo_drop_deepseek", "loo_drop_llama",
               "loo_drop_qwen", "loo_drop_openai", "loo_drop_anthropic"]
    eval_keys = EVAL_COLS

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(recipes))
    w = 0.20

    colors_eval = [NAVY, TEAL, "#9c6f3a", CORAL]
    for j, (k, lab, src) in enumerate(eval_keys):
        vals = [acc(r, k, src) * 100 for r in recipes]
        offset = (j - 1.5) * w
        bars = ax.bar(x + offset, vals, w, color=colors_eval[j],
                      label=lab.replace("\n", " "))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([RECIPE_LABEL[r] for r in recipes], rotation=18, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Leave-one-out ablation: which generators matter?")
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)

    fig.text(0.02, -0.06,
             "Dropping Anthropic Haiku collapses every eval set (60/57/42/60). "
             "Dropping DeepSeek matches or beats the full mix - it is redundant.",
             fontsize=9, color="#444")

    fig.tight_layout()
    fig.savefig(OUT / "3_loo_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 4
def fig_layer_probe_auc():
    layers = list(range(13))
    baseline = [STAGE2_B["by_layer"][L]["linear_eq"]["manual"] for L in layers]
    multi    = [STAGE2_M["by_layer"][L]["linear_eq"]["manual"] for L in layers]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(layers, baseline, "-o", color=CORAL, linewidth=2.2,
            markersize=6, label="baseline (single-gen)")
    ax.plot(layers, multi,    "-o", color=NAVY,  linewidth=2.2,
            markersize=6, label="multi-generator (mitigation)")

    ax.axvspan(9.5, 12.5, color="#fbecea", alpha=0.7, zorder=0)
    ax.text(11, 0.55, "shortcut region\n(baseline only)",
            ha="center", va="center", fontsize=9, color=CORAL, alpha=0.9)

    ax.set_xticks(layers)
    ax.set_xlabel("transformer layer (CLS residual)")
    ax.set_ylabel("manual eval AUC (linear probe)")
    ax.set_ylim(0.4, 1.04)
    ax.set_title("Where the shortcut lives: per-layer probe AUC on humans")
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    # annotate baseline's final-layer drop
    for L in [10, 11, 12]:
        ax.annotate(f"{baseline[L]:.2f}",
                    xy=(L, baseline[L]), xytext=(L, baseline[L] - 0.07),
                    ha="center", fontsize=8.5, color=CORAL)
    for L in [10, 11, 12]:
        ax.annotate(f"{multi[L]:.2f}",
                    xy=(L, multi[L]), xytext=(L, multi[L] + 0.025),
                    ha="center", fontsize=8.5, color=NAVY)

    fig.text(0.02, -0.04,
             "Equivalence signal peaks at layer 10 in both models. "
             "Only the baseline degrades at layers 11-12 - that is the shortcut overwriting semantics.",
             fontsize=9, color="#444")

    fig.tight_layout()
    fig.savefig(OUT / "4_layer_probe_auc.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 5
def fig_direction_ablation():
    sweep = STAGE6_B["sweep"]
    Ks    = [s["K"] for s in sweep]
    auc_m = [s["eq_manual"] for s in sweep]

    # cumulative direction labels shown directly on the x-axis
    added = ["(none)", "neg-flip", "+ length-ratio", "+ 'the post' prefix",
             "+ first-person", "+ much-shorter", "+ verdict-word"]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    # baseline reference band
    base = auc_m[0]
    ax.axhspan(base - 0.005, base + 0.005, color="#f4f6f8", zorder=0)
    ax.axhline(base, color=SLATE, linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=1, label=f"K=0 baseline ({base:.3f})")

    # bars
    bars = ax.bar(Ks, auc_m, width=0.55, color=CORAL,
                  edgecolor="white", linewidth=1.2, zorder=2)
    for b, v in zip(bars, auc_m):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")

    ax.set_xticks(Ks)
    ax.set_xticklabels([f"K={k}\n{added[k]}" for k in Ks], fontsize=9.5)
    ax.set_xlabel("directions ablated at layer 12 (cumulative)")
    ax.set_ylabel("manual eval AUC (baseline model)")
    ax.set_ylim(0.72, 0.81)
    ax.set_yticks([0.72, 0.74, 0.76, 0.78, 0.80])
    ax.set_title("6-direction ablation: the shortcut is redundantly encoded")
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    fig.text(0.02, 0.005,
             "Projecting out six hand-picked stylistic directions at L12 moves manual AUC by 0.011. "
             "The shortcut cannot be removed at inference; it must be prevented at training.",
             fontsize=9, color="#444")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "5_direction_ablation.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- 6
def fig_headline_efficiency():
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def panel(x0, x1, color, title, pairs, footer):
        # title bar
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (x0, 0.4), x1 - x0, 5.1,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.6, edgecolor=color, facecolor="white"))
        ax.text((x0+x1)/2, 4.95, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
        # body lines
        for i, (k, v) in enumerate(pairs):
            y = 4.0 - i * 1.0
            ax.text(x0 + 0.3, y, k, fontsize=11, color="#444", va="center")
            ax.text(x1 - 0.3, y, v, fontsize=14, color=color,
                    fontweight="bold", ha="right", va="center")
        ax.text((x0+x1)/2, 0.85, footer, ha="center", va="center",
                fontsize=9.5, color="#444", style="italic")

    panel(0.2, 4.6, SLATE, "single-generator baseline",
          [("training pairs",   "52,137"),
           ("handwritten acc",  "55%"),
           ("eq / uneq split",  "31 / 100"),
           ("Anthropic Haiku eval", "59%")],
          "Qwen-72B prose, large corpus")

    panel(5.4, 9.8, CORAL, "multi-generator recipe",
          [("training pairs",   "800"),
           ("handwritten acc",  "95%"),
           ("eq / uneq split",  "92 / 100"),
           ("Anthropic Haiku eval", "81%")],
          "4 LLMs x 200 pairs (Haiku, GPT, Qwen, Llama)")

    # arrow
    ax.annotate("", xy=(5.25, 3.0), xytext=(4.75, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8))
    ax.text(5.0, 3.45, "65x less data\n+40 pp acc",
            ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

    fig.suptitle("Generator diversity beats data scale",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT / "6_headline_efficiency.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# v2 BATCH
# ============================================================================

def _heat(metric_key: str, title: str, fname: str, vrange=(0.40, 1.0),
          fmt=lambda v: f"{v*100:.0f}"):
    rows = ROW_ORDER
    cols = [c[0] for c in EVAL_COLS]
    M = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, (k, _, src) in enumerate(EVAL_COLS):
            v = src[r][k].get(metric_key, np.nan)
            M[i, j] = v if v is not None else np.nan

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "slatecoral", ["#f4f6f8", "#cfd6df", "#7d8a99", CORAL])
    norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in EVAL_COLS], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([RECIPE_LABEL[r] for r in rows], fontsize=10)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Evaluation set (human-style writing)")
    ax.set_title(title)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            color = "white" if (v - vrange[0]) / (vrange[1] - vrange[0]) >= 0.55 else INK
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
    for y in [0.5, 5.5, 10.5]:
        ax.axhline(y, color="white", linewidth=2)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


# 7 -------------------------------------------------------------------- hate
def fig_hate_classifier():
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    groups   = ["Davidson 2017\n(n=4957)", "HateXplain\n(n=3846)"]
    roberta  = [STAGE4_DR["accuracy"], STAGE4_HR["accuracy"]]
    tfidf    = [STAGE4_DT["accuracy"], STAGE4_HT["accuracy"]]
    f1_rob   = [STAGE4_DR["f1_macro"],  STAGE4_HR["f1_macro"]]
    f1_tf    = [STAGE4_DT["f1_macro"],  STAGE4_HT["f1_macro"]]

    x = np.arange(len(groups))
    w = 0.20
    b1 = ax.bar(x - 1.5*w, [v*100 for v in roberta], w, color=NAVY,  label="RoBERTa-base acc")
    b2 = ax.bar(x - 0.5*w, [v*100 for v in f1_rob],  w, color="#6f8eb8", label="RoBERTa F1-macro")
    b3 = ax.bar(x + 0.5*w, [v*100 for v in tfidf],   w, color=SLATE,  label="TF-IDF + LR acc")
    b4 = ax.bar(x + 1.5*w, [v*100 for v in f1_tf],   w, color="#aab5bf", label="TF-IDF F1-macro")

    for bars, vals in [(b1, roberta), (b2, f1_rob), (b3, tfidf), (b4, f1_tf)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0,
                    f"{v*100:.1f}", ha="center", va="bottom",
                    fontsize=9, color=INK)

    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("score (%)")
    ax.set_title("Hate-speech classifier baselines (3-class)")
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)

    fig.text(0.02, -0.04,
             "Sanity-check baselines: RoBERTa beats TF-IDF on both datasets, "
             "matching paper numbers. Not the rationale-judge task.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "7_hate_classifier.png", bbox_inches="tight")
    plt.close(fig)


# 8 -------------------------------------------------------------- training dyn
def fig_training_dynamics():
    tl = STAGE3["timeline"]
    steps = [t["step"] for t in tl]
    ood   = [t["ood_acc"] * 100 for t in tl]
    man   = [t["manual_acc"] * 100 for t in tl]
    cl    = [t["claude_acc"] * 100 for t in tl]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(steps, ood, "-o", color=NAVY,  linewidth=2.0, markersize=5, label="OOD synthetic (n~10k)")
    ax.plot(steps, cl,  "-o", color=TEAL,  linewidth=2.0, markersize=5, label="Claude 4.6 eval (n=20)")
    ax.plot(steps, man, "-o", color=CORAL, linewidth=2.2, markersize=6, label="manual handwritten (n=20)")

    ax.set_xlabel("training step (baseline, batch=16)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(30, 105)
    ax.set_title("Training dynamics: OOD acc climbs, human acc bounces")
    ax.legend(loc="lower right", frameon=False, fontsize=10)

    fig.text(0.02, -0.04,
             "Synthetic OOD accuracy plateaus near 97% within ~500 steps. "
             "Manual eval is high-variance and uncorrelated - it cannot be used for model selection.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "8_training_dynamics.png", bbox_inches="tight")
    plt.close(fig)


# 9 ---------------------------------------------------------------- AUROC heat
def fig_auroc_matrix():
    _heat("auroc", "Training recipe x evaluation set: AUROC",
          "9_auroc_matrix.png",
          vrange=(0.10, 1.0), fmt=lambda v: f"{v:.2f}")


# 10 -------------------------------------------------------------------- F1
def fig_f1_matrix():
    _heat("f1", "Training recipe x evaluation set: F1-macro",
          "10_f1_matrix.png",
          vrange=(0.30, 1.0), fmt=lambda v: f"{v:.2f}")


# 11 ------------------------------------------------------- AUROC vs acc scatter
def fig_auroc_vs_acc():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for r in ROW_ORDER:
        a = STAGE14B[r]["manual_handwritten_n20"]["acc"]   * 100
        u = STAGE14B[r]["manual_handwritten_n20"]["auroc"] * 100
        is_winner = r in ("multi_gen", "loo_drop_deepseek")
        is_loo    = r.startswith("loo_drop_") and not is_winner
        is_single = r in ("anthropic_haiku","openai_gpt4omini","qwen25_72b",
                          "meta_llama3_8b","deepseek_v3")
        color = CORAL if is_winner else (NAVY if is_loo else (SLATE if is_single else "#777"))
        size  = 140 if is_winner else 90
        ax.scatter(a, u, s=size, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        # label
        dx, dy = 1.5, -2.5
        if r == "multi_gen":      dx, dy = -3, 3
        if r == "poster_baseline": dx, dy = -3, 3
        ax.text(a + dx, u + dy, RECIPE_LABEL[r], fontsize=8.5, color=INK)

    ax.axhline(50, color="#bbb", linewidth=0.6, zorder=1)
    ax.axvline(50, color="#bbb", linewidth=0.6, zorder=1)
    ax.fill_between([0, 70], 80, 100, color="#fdf1ee", alpha=0.5, zorder=0)
    ax.text(35, 95, "high AUROC + low acc\n= collapsed classifier",
            fontsize=10, color=CORAL, ha="center", style="italic")

    ax.set_xlim(20, 105); ax.set_ylim(20, 105)
    ax.set_xlabel("handwritten n=20 accuracy (%)")
    ax.set_ylabel("handwritten n=20 AUROC (%)")
    ax.set_title("AUROC vs Accuracy on handwritten test set")
    fig.text(0.02, -0.02,
             "The collapsed single-generators sit upper-left: the ranking is fine, "
             "but the threshold is broken. multi-gen sits upper-right.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "11_auroc_vs_acc.png", bbox_inches="tight")
    plt.close(fig)


# 12 ----------------------------------------------------- eq vs neg probes layer
def fig_eq_vs_neg_probe():
    by = PROBE_M["metrics_by_layer"]
    layers = [r["layer"] for r in by]
    eq_val   = [r["probe_equiv_auc_val"]  for r in by]
    neg_val  = [r["probe_neg_auc_val"]    for r in by]
    eq_man   = [r["probe_equiv_auc_manual"] for r in by]
    neg_man  = [r["probe_neg_auc_manual"]   for r in by]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(layers, eq_val,  "-o", color=NAVY,  linewidth=2.0, label="equivalence probe (synthetic-val)")
    ax.plot(layers, neg_val, "-o", color=TEAL,  linewidth=2.0, label="negation probe (synthetic-val)")
    ax.plot(layers, eq_man,  "--s", color=CORAL, linewidth=2.0, label="equivalence probe (manual)")
    ax.plot(layers, neg_man, "--s", color="#9c6f3a", linewidth=2.0, label="negation probe (manual)")

    ax.set_xlabel("transformer layer (CLS residual)")
    ax.set_ylabel("linear probe AUC")
    ax.set_xticks(layers)
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Equivalence and negation are linearly encoded together")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.text(0.02, -0.04,
             "Both probes climb together. Negation correlates with equivalence in synthetic data; "
             "the model exploits that on OOD but collapses on handwritten where the correlation breaks.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "12_eq_vs_neg_probes.png", bbox_inches="tight")
    plt.close(fig)


# 13 ----------------------------------------------------- mitigation journey
def fig_mitigation_journey():
    order = ["v00_baseline", "v01_drop_high_cosine", "v02_drop_neg_shortcut",
             "v03_balance_neg", "v05_one_epoch", "v06_combined",
             "v04_style_mix_qwen7b"]
    label = {
        "v00_baseline":        "v0: baseline",
        "v01_drop_high_cosine":"v1: drop high-cos",
        "v02_drop_neg_shortcut":"v2: drop neg shortcut",
        "v03_balance_neg":     "v3: balance negation",
        "v05_one_epoch":       "v5: 1 epoch",
        "v06_combined":        "v6: combined",
        "v04_style_mix_qwen7b":"v4: style mix (best)",
    }
    ood  = [STAGE1[k]["ood_overall"]["accuracy"] * 100 for k in order]
    man  = [STAGE1[k]["manual"]["accuracy"]      * 100 for k in order]
    cl   = [STAGE1[k]["claude"]["accuracy"]      * 100 for k in order]
    # plus multi-gen from stage14b
    mg_man = STAGE14B["multi_gen"]["manual_handwritten_n20"]["acc"] * 100
    mg_cl  = STAGE14B["multi_gen"]["claude46_n20"]["acc"] * 100

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(order) + 1)
    w = 0.27
    labels = [label[k] for k in order] + ["multi-gen\n(stage 14)"]
    b1 = ax.bar(x - w, ood + [np.nan], w, color="#a8b4c2", label="OOD synthetic (val)")
    b2 = ax.bar(x,     man + [mg_man], w, color=CORAL,     label="manual handwritten")
    b3 = ax.bar(x + w, cl  + [mg_cl],  w, color=NAVY,      label="Claude 4.6")

    for bars, vals in [(b1, ood + [np.nan]), (b2, man + [mg_man]), (b3, cl + [mg_cl])]:
        for b, v in zip(bars, vals):
            if np.isnan(v): continue
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 110); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Mitigation journey: every recipe we tried")
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=3)

    fig.text(0.02, -0.04,
             "Single-direction mitigations (drop high-cos, drop neg) barely moved manual. "
             "Style mixing (v4) was the first to crack it; multi-gen made it robust.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "13_mitigation_journey.png", bbox_inches="tight")
    plt.close(fig)


# 14 ----------------------------------------------------------- LLM ceiling
def fig_llm_ceiling():
    nice = {
        "anthropic/claude-sonnet-4.5":  "Claude Sonnet 4.5",
        "openai/gpt-4o-2024-11-20":     "GPT-4o",
        "anthropic/claude-haiku-4.5":   "Claude Haiku 4.5",
        "qwen/qwen-2.5-72b-instruct":   "Qwen 2.5 72B",
        "deepseek/deepseek-chat-v3.1":  "DeepSeek V3",
    }
    order = ["anthropic/claude-sonnet-4.5", "openai/gpt-4o-2024-11-20",
             "anthropic/claude-haiku-4.5",  "qwen/qwen-2.5-72b-instruct",
             "deepseek/deepseek-chat-v3.1"]
    acc = [STAGE11[k]["accuracy"] * 100 for k in order]
    # multi-gen reference on stage10_haiku n=100 = 81%
    mg = STAGE14["multi_gen"]["stage10_haiku_n100"]["acc"] * 100
    base = STAGE14["poster_baseline"]["stage10_haiku_n100"]["acc"] * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(order) + 2)
    labels = [nice[k] for k in order] + ["multi-gen\nroberta-base", "baseline\nroberta-base"]
    vals   = acc + [mg, base]
    colors = [NAVY] * len(order) + [CORAL, SLATE]
    bars = ax.bar(x, vals, color=colors, width=0.65)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("accuracy on Claude-Haiku reactions (n=100)")
    ax.set_title("Frontier LLM ceiling vs our 125M RoBERTa")

    ax.axhline(mg, color=CORAL, linewidth=0.8, linestyle="--", alpha=0.6)
    fig.text(0.02, -0.04,
             "Frontier LLMs ceiling at ~99-100%. Our multi-gen RoBERTa-base hits 81% with <1% the inference cost.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "14_llm_ceiling.png", bbox_inches="tight")
    plt.close(fig)


# 15 -------------------------------------------------- sentence-BERT baseline
def fig_sentence_bert():
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    evals  = [("stage10_human_n100", "Haiku\nn=100", "stage10_haiku_n100"),
              ("stage7_gpt_n50",     "GPT-mini\nn=50", "stage7_gpt_n50"),
              ("manual_plus_claude_n40", "Manual+Claude\nn=40", "manual_plus_claude_n40")]
    cos = [STAGE13[k]["cosine_threshold"]["acc"] * 100 for k, _, _ in evals]
    log = [STAGE13[k]["logreg_3feat"]["acc"]    * 100 for k, _, _ in evals]
    mg  = [STAGE14["multi_gen"][m]["acc"]       * 100 for _, _, m in evals]
    base= [STAGE14["poster_baseline"][m]["acc"] * 100 for _, _, m in evals]

    x = np.arange(len(evals)); w = 0.20
    b1 = ax.bar(x - 1.5*w, cos,  w, color=SLATE, label="SBERT cosine + threshold")
    b2 = ax.bar(x - 0.5*w, log,  w, color="#a8b4c2", label="SBERT + LogReg (3 features)")
    b3 = ax.bar(x + 0.5*w, base, w, color="#cba293", label="baseline RoBERTa")
    b4 = ax.bar(x + 1.5*w, mg,   w, color=CORAL, label="multi-gen RoBERTa")
    for bars, vals in [(b1, cos), (b2, log), (b3, base), (b4, mg)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([e[1] for e in evals])
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Off-the-shelf SentenceBERT vs fine-tuned RoBERTa")
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    fig.text(0.02, -0.04,
             "Pretrained semantic similarity is not enough: SBERT can't beat 50%. "
             "Fine-tuning on the right data is what carries the signal.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "15_sentence_bert.png", bbox_inches="tight")
    plt.close(fig)


# 16 ---------------------------------------------- per-class for ALL 12 recipes
def fig_per_class_all():
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    rows = ROW_ORDER
    eq   = [STAGE14B[r]["manual_handwritten_n20"]["acc_eq"]   * 100 for r in rows]
    uneq = [STAGE14B[r]["manual_handwritten_n20"]["acc_uneq"] * 100 for r in rows]
    x = np.arange(len(rows)); w = 0.38
    b1 = ax.bar(x - w/2, eq,   w, color=NAVY,  label="equivalent class")
    b2 = ax.bar(x + w/2, uneq, w, color=CORAL, label="unequivalent class")
    for bars, vals in [(b1, eq), (b2, uneq)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.2,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([RECIPE_LABEL[r] for r in rows],
                                          rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("per-class accuracy (%)")
    ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=10)
    ax.set_title("Per-class accuracy on handwritten test set (n=20) - all 12 recipes")
    for sep in [0.5, 5.5, 10.5]:
        ax.axvline(sep, color="#ddd", linewidth=0.8, linestyle=":")
    fig.text(0.02, -0.04,
             "From left: baseline, 5 single-gens, 5 leave-one-outs, multi-gen. "
             "Look for a recipe with both bars near 100 - only the right edge qualifies.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "16_per_class_all.png", bbox_inches="tight")
    plt.close(fig)


# 17 ----------------------------------------------------- fine training sweep
def fig_fine_training_sweep():
    tl = STAGE5["timeline"]
    steps = [t["step"] for t in tl]
    ood   = [t["ood_acc"]   * 100 for t in tl]
    man   = [t["manual_acc"] * 100 for t in tl]
    cl    = [t["claude_acc"] * 100 for t in tl]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(steps, ood, "-o", color=NAVY,  linewidth=2.0, markersize=4, label="OOD synthetic")
    ax.plot(steps, cl,  "-o", color=TEAL,  linewidth=2.0, markersize=4, label="Claude 4.6 eval")
    ax.plot(steps, man, "-o", color=CORAL, linewidth=2.2, markersize=5, label="manual handwritten")
    ax.set_xlabel("training step (fine-grained, 25-step intervals)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(20, 105)
    ax.set_title("Fine-grained training sweep: manual eval is noise")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.text(0.02, -0.04,
             "OOD plateaus at 96-97% by step 100. Manual bounces in [30, 80] for the rest of training. "
             "You cannot use a synthetic val set to select a checkpoint that generalises to humans.",
             fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "17_fine_training_sweep.png", bbox_inches="tight")
    plt.close(fig)


# 18 -------------------------------------------- multi-gen 4-eval scorecard
def fig_multigen_scorecard():
    evals = [
        ("Handwritten (n=20)",     STAGE14B["multi_gen"]["manual_handwritten_n20"]),
        ("Claude 4.6 (n=20)",      STAGE14B["multi_gen"]["claude46_n20"]),
        ("GPT-4o-mini (n=50)",     STAGE14["multi_gen"]["stage7_gpt_n50"]),
        ("Claude Haiku (n=100)",   STAGE14["multi_gen"]["stage10_haiku_n100"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8))
    for ax, (name, d) in zip(axes, evals):
        a   = d["acc"]   * 100
        au  = d["auroc"] * 100
        eq  = d["acc_eq"]   * 100
        uneq= d["acc_uneq"] * 100
        ax.bar([0, 1, 2, 3], [a, au, eq, uneq],
               color=[CORAL, NAVY, "#6f8eb8", "#9c6f3a"], width=0.7)
        for i, v in enumerate([a, au, eq, uneq]):
            ax.text(i, v + 2, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["acc", "AUROC", "eq", "uneq"], fontsize=9)
        ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(name, fontsize=11)
        ax.grid(True, axis="y", color="#eef0f2", linewidth=0.5)
    fig.suptitle("Multi-generator model: scorecard across all four human-style evals",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "18_multigen_scorecard.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_eval_matrix()
    fig_per_class_balance()
    fig_loo_ablation()
    fig_layer_probe_auc()
    fig_direction_ablation()
    fig_headline_efficiency()
    # v2 batch
    fig_hate_classifier()
    fig_training_dynamics()
    fig_auroc_matrix()
    fig_f1_matrix()
    fig_auroc_vs_acc()
    fig_eq_vs_neg_probe()
    fig_mitigation_journey()
    fig_llm_ceiling()
    fig_sentence_bert()
    fig_per_class_all()
    fig_fine_training_sweep()
    fig_multigen_scorecard()
    print(f"wrote 18 figures to {OUT}")
