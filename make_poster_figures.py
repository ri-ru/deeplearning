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

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(Ks, auc_m, "-o", color=CORAL, linewidth=2.2, markersize=8)

    for K, v in zip(Ks, auc_m):
        ax.annotate(f"{v:.3f}", xy=(K, v), xytext=(K, v + 0.008),
                    ha="center", fontsize=9, color=INK)

    ax.set_xlabel("K = number of stylistic directions ablated at layer 12")
    ax.set_ylabel("manual eval AUC (baseline)")
    ax.set_ylim(0.72, 0.82)
    ax.set_xticks(Ks)
    ax.set_title("6-direction ablation: the shortcut is redundantly encoded")

    directions = ["neg-flip", "+length", "+'the post'", "+1st-person", "+shorter", "+verdict"]
    ann = "directions ablated, cumulative:\n  K=1: neg-flip\n  K=2: + length-ratio\n  K=3: + 'the post' prefix\n  K=4: + first-person\n  K=5: + much-shorter\n  K=6: + verdict-word"
    ax.text(0.02, 0.02, ann, transform=ax.transAxes,
            fontsize=8.5, color="#444", va="bottom",
            family="monospace",
            bbox=dict(facecolor="#f4f6f8", edgecolor="none", pad=8))

    fig.text(0.02, -0.04,
             "Removing six hand-picked stylistic directions at L12 moves manual AUC by 0.011. "
             "The shortcut cannot be removed at inference; it must be prevented at training.",
             fontsize=9, color="#444")

    fig.tight_layout()
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


if __name__ == "__main__":
    fig_eval_matrix()
    fig_per_class_balance()
    fig_loo_ablation()
    fig_layer_probe_auc()
    fig_direction_ablation()
    fig_headline_efficiency()
    print(f"wrote 6 figures to {OUT}")
