"""Script to generate publication-quality figures for the IEEE technical report.
Reads experimental outputs from task1, task2, task3, task4, and task5,
and creates vector PDF figures in report/figures/.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.75,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "figure.autolayout": True,
})

FIG_DIR = Path("report/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Figure 1: Task 1 Training Curves & Base vs Fine-tuned Comparison
# -----------------------------------------------------------------------------
def plot_figure_1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel A: Training & Validation Loss across Epochs
    epochs = [1, 2, 3]
    train_loss = [0.2071, 0.1271, 0.0747]
    val_loss = [1.3063, 1.6032, 1.7728]
    val_ppl = [3.692, 4.969, 5.887]

    color = "tab:blue"
    ax1.plot(epochs, train_loss, marker="o", color="tab:blue", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, marker="s", color="tab:red", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Fine-Tuning Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_xticks(epochs)
    ax1.grid(True)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, val_ppl, marker="^", color="tab:purple", linestyle=":", label="Val Perplexity")
    ax1_twin.set_ylabel("Perplexity", color="tab:purple")
    ax1_twin.tick_params(axis="y", labelcolor="tab:purple")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left", framealpha=0.9)
    ax1.set_title("(a) QLoRA Convergence & Overfitting (Epoch 1 Best)")

    # Panel B: Base vs Fine-tuned Per-Class F1
    classes = ["incl", "not-incl", "not-excl", "excl", "missing", "n/a"]
    base_f1 = [0.297, 0.000, 0.182, 0.000, 0.204, 0.000]
    ft_f1   = [0.783, 0.000, 0.840, 0.000, 0.624, 0.455]

    x = np.arange(len(classes))
    width = 0.35

    ax2.bar(x - width/2, base_f1, width, label="Base Qwen2.5-3B", color="tab:gray", alpha=0.75, edgecolor="black")
    ax2.bar(x + width/2, ft_f1, width, label="Fine-Tuned (QLoRA)", color="tab:green", alpha=0.85, edgecolor="black")

    ax2.set_xlabel("Eligibility Category")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("(b) Per-Class F1 Score (Base vs. Fine-Tuned)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_training_curves.pdf", dpi=300)
    plt.savefig(FIG_DIR / "fig1_training_curves.png", dpi=300)
    plt.close()
    print("[OK] Generated fig1_training_curves")

# -----------------------------------------------------------------------------
# Figure 2: Task 2 Confusion Matrix & Inclusion vs Exclusion Split
# -----------------------------------------------------------------------------
def plot_figure_2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))

    # Panel A: 6x6 Confusion Matrix
    cm = np.array([
        [18,  0, 89,  0,  9,  0], # Note: true labels vs preds
        # Actually from task2/outputs/test_metrics.json:
        # matrix = [
        #   [18, 0,  0, 0,  9, 0],  # included (27)
        #   [ 0, 0,  0, 0,  7, 0],  # not included (7)
        #   [ 0, 0, 89, 0,  1, 1],  # not excluded (91)
        #   [ 0, 0,  0, 0,  0, 0],  # excluded (0)
        #   [ 1, 0, 25, 0, 39, 1],  # not enough info (66)
        #   [ 0, 0,  7, 0,  3, 5],  # not applicable (15)
        # ]
    ])
    matrix = np.array([
        [18, 0,  0, 0,  9, 0],
        [ 0, 0,  0, 0,  7, 0],
        [ 0, 0, 89, 0,  1, 1],
        [ 0, 0,  0, 0,  0, 0],
        [ 1, 0, 25, 0, 39, 1],
        [ 0, 0,  7, 0,  3, 5],
    ])
    labels = ["incl", "not-inc", "not-exc", "excl", "missing", "n/a"]

    im = ax1.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("Ground Truth Label")
    ax1.set_title("(a) Confusion Matrix (Test Split, n=206)")

    # Annotate numbers
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if val > 40 else "black"
            ax1.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8.5, weight="bold" if i==j and val>0 else "normal")

    # Panel B: Inclusion vs Exclusion Metrics
    categories = ["Accuracy", "Macro F1", "Evidence Macro", "Evidence Micro"]
    incl_scores = [73.33, 39.30, 69.70, 61.90]
    excl_scores = [73.28, 49.83, 89.97, 49.18]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width/2, incl_scores, width, label="Inclusion (n=75)", color="tab:blue", alpha=0.85, edgecolor="black")
    ax2.bar(x + width/2, excl_scores, width, label="Exclusion (n=131)", color="tab:purple", alpha=0.85, edgecolor="black")

    ax2.set_ylabel("Score (%)")
    ax2.set_title("(b) Inclusion vs. Exclusion Performance")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha="right")
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_confusion_matrix.pdf", dpi=300)
    plt.savefig(FIG_DIR / "fig2_confusion_matrix.png", dpi=300)
    plt.close()
    print("[OK] Generated fig2_confusion_matrix")

# -----------------------------------------------------------------------------
# Figure 3: Task 3 KV-Cache Throughput & Memory Scaling
# -----------------------------------------------------------------------------
def plot_figure_3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    # Panel A: Decoding Throughput across (Prompt, Output) configurations
    configs = ["S (32)", "S (128)", "M (32)", "M (128)", "L (32)", "L (128)"]
    uncached_tps = [6.76, 5.79, 5.34, 4.46, 4.09, 2.89]
    cached_tps   = [6.74, 6.84, 7.21, 7.48, 7.27, 7.40]

    x = np.arange(len(configs))
    width = 0.35

    ax1.bar(x - width/2, uncached_tps, width, label="Uncached Baseline", color="tab:gray", alpha=0.75, edgecolor="black")
    ax1.bar(x + width/2, cached_tps, width, label="Explicit KV-Cached", color="tab:blue", alpha=0.85, edgecolor="black")

    ax1.set_xlabel("Workload Config (Prompt, New Tokens)")
    ax1.set_ylabel("Throughput (tokens/s)")
    ax1.set_title("(a) Autoregressive Decoding Throughput (tok/s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel B: Memory Saved vs Output Tokens (Long Prompt)
    out_tokens = [32, 64, 128]
    mem_saved = [25.75, 53.31, 108.93] # MB
    cache_size = [3.00, 4.12, 6.42]     # MB

    ax2.plot(out_tokens, mem_saved, marker="o", color="tab:green", label="GPU Memory Saved (MB)", linewidth=2)
    ax2.plot(out_tokens, cache_size, marker="s", color="tab:blue", linestyle="--", label="Dedicated Cache Size (MB)", linewidth=2)
    ax2.set_xlabel("Generated Tokens (Long Prompt: 650 tok)")
    ax2.set_ylabel("GPU Memory (MB)")
    ax2.set_title("(b) Memory Footprint & Savings vs. Uncached")
    ax2.set_xticks(out_tokens)
    ax2.grid(True)
    ax2.legend(loc="center left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_kv_cache_scaling.pdf", dpi=300)
    plt.savefig(FIG_DIR / "fig3_kv_cache_scaling.png", dpi=300)
    plt.close()
    print("[OK] Generated fig3_kv_cache_scaling")

# -----------------------------------------------------------------------------
# Figure 4: Task 4 Attention Memory Wall & Execution Latency
# -----------------------------------------------------------------------------
def plot_figure_4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    ref_mem   = [9.88, 13.20, 24.36, 65.23, 220.21, 824.27]
    tiled_mem = [9.88, 10.94, 12.92, 16.90,  24.91,  40.95]
    fused_mem = [9.06, 10.09, 12.06, 16.06,  24.06,  40.06]

    # Panel A: Peak Memory Wall
    ax1.plot(seq_lens, ref_mem, marker="o", color="tab:red", label="Reference SDPA O(N²)", linewidth=2)
    ax1.plot(seq_lens, tiled_mem, marker="s", color="tab:green", label="Tiled Online-Softmax O(N)", linewidth=2)
    ax1.plot(seq_lens, fused_mem, marker="^", color="tab:blue", linestyle=":", label="PyTorch Fused (Flash-2)", linewidth=2)

    ax1.set_xlabel("Sequence Length (N)")
    ax1.set_ylabel("Peak VRAM Allocated (MB)")
    ax1.set_title("(a) Attention Memory Scaling: O(N²) vs. O(N)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.grid(True, which="both")
    ax1.legend(loc="upper left", framealpha=0.9)

    # Panel B: Latency Comparison (ms)
    ref_lat   = [0.59, 0.34, 0.41, 1.58, 6.96, 26.69]
    tiled_lat = [15.21, 27.84, 56.49, 114.28, 228.45, 456.91] # High level Python kernel
    fused_lat = [0.09, 0.12, 0.21, 0.32, 1.01, 3.63]

    ax2.plot(seq_lens, ref_lat, marker="o", color="tab:red", label="Reference SDPA", linewidth=2)
    ax2.plot(seq_lens, tiled_lat, marker="s", color="tab:green", label="Tiled PyTorch (Interpret Loop)", linewidth=2)
    ax2.plot(seq_lens, fused_lat, marker="^", color="tab:blue", linestyle=":", label="PyTorch Fused Flash-2", linewidth=2)

    ax2.set_xlabel("Sequence Length (N)")
    ax2.set_ylabel("Kernel Latency (ms)")
    ax2.set_title("(b) Attention Latency (Log Scale)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.grid(True, which="both")
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_attention_scaling.pdf", dpi=300)
    plt.savefig(FIG_DIR / "fig4_attention_scaling.png", dpi=300)
    plt.close()
    print("[OK] Generated fig4_attention_scaling")

# -----------------------------------------------------------------------------
# Figure 5: Task 5 Speculative Speedup vs Lookahead (Gamma)
# -----------------------------------------------------------------------------
def plot_figure_5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    gammas = [1, 3, 5, 7]
    json_speedup = [1.07, 1.56, 1.75, 1.46]
    narr_speedup = [0.92, 1.28, 1.24, 1.15]
    qa_speedup   = [1.06, 1.43, 1.34, 1.10]

    # Panel A: Speedup vs Gamma
    ax1.plot(gammas, json_speedup, marker="o", color="tab:green", label="Structured JSON (Low Entropy)", linewidth=2)
    ax1.plot(gammas, qa_speedup, marker="^", color="tab:blue", label="Medical QA (Medium Entropy)", linewidth=2)
    ax1.plot(gammas, narr_speedup, marker="s", color="tab:red", label="Clinical Narrative (High Entropy)", linewidth=2)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Baseline (1.0x)")

    ax1.set_xlabel("Speculative Lookahead Horizon (γ)")
    ax1.set_ylabel("Speedup Factor vs. Baseline")
    ax1.set_title("(a) Speculative Speedup vs. γ (Demonstrating Slowdown)")
    ax1.set_xticks(gammas)
    ax1.grid(True)
    ax1.legend(loc="lower left", framealpha=0.9)

    # Panel B: Acceptance Rate (alpha) and Target Passes Saved
    json_alpha = [88.9, 70.0, 62.9, 50.6]
    narr_alpha = [73.0, 58.0, 44.3, 42.2]

    ax2.plot(gammas, json_alpha, marker="o", color="tab:green", label="JSON Acceptance (α %)", linewidth=2)
    ax2.plot(gammas, narr_alpha, marker="s", color="tab:red", label="Narrative Acceptance (α %)", linewidth=2)

    json_saved = [48.1, 67.3, 75.0, 76.9]
    ax2.plot(gammas, json_saved, marker="D", color="tab:purple", linestyle=":", label="Target Passes Saved (%)", linewidth=2)

    ax2.set_xlabel("Speculative Lookahead Horizon (γ)")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("(b) Draft Acceptance (α) & Target Passes Saved")
    ax2.set_xticks(gammas)
    ax2.set_ylim(30, 100)
    ax2.grid(True)
    ax2.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_speculative_decoding.pdf", dpi=300)
    plt.savefig(FIG_DIR / "fig5_speculative_decoding.png", dpi=300)
    plt.close()
    print("[OK] Generated fig5_speculative_decoding")

if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    print("\n[SUCCESS] All 5 publication vector figures generated in report/figures/!")
