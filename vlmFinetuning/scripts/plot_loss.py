#!/usr/bin/env python3
"""Plot training loss vs validation loss from a run's metrics.jsonl (written by train.py).

  python scripts/plot_loss.py runs/qwen25vl_dryrun_B4_e5
  python scripts/plot_loss.py runs/qwen25vl_dryrun_B4_e5 runs/qwen25vl_dryrun_B8_e5     # one plot per run + overlay

Writes <run>/loss_plot.png (and runs/loss_comparison.png when several runs are given) and prints the loss table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display needed on the server
import matplotlib.pyplot as plt  # noqa: E402


def read_losses(run: Path):
    path = run / "metrics.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} not found (has the run started training?)")
    train, val, steps_per_epoch = {}, {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        step, epoch = m.get("step"), m.get("epoch")
        if "loss" in m:                    # training loss (resumed runs may repeat a step: keep the last)
            train[step] = (epoch, float(m["loss"]))
        if "eval_loss" in m:
            val[step] = (epoch, float(m["eval_loss"]))
        if step and epoch:
            steps_per_epoch = step / epoch
    return sorted(train.items()), sorted(val.items()), steps_per_epoch


def print_table(name: str, train, val) -> None:
    t, v = dict(train), dict(val)
    print(f"\n{name}")
    print(f"{'step':>5} {'epoch':>6} {'train_loss':>11} {'val_loss':>9}")
    for step in sorted(set(t) | set(v)):
        epoch = (t.get(step) or v.get(step))[0]
        tl = f"{t[step][1]:.4f}" if step in t else ""
        vl = f"{v[step][1]:.4f}" if step in v else ""
        print(f"{step:5d} {epoch:6.2f} {tl:>11} {vl:>9}")
    if train:
        print(f"train loss: first {train[0][1][1]:.4f} -> last {train[-1][1][1]:.4f}")
    if val:
        best = min(val, key=lambda x: x[1][1])
        print(f"val loss:   first {val[0][1][1]:.4f} -> last {val[-1][1][1]:.4f} | best {best[1][1]:.4f} "
              f"at step {best[0]} (kept as final_adapter)")


def draw(ax, train, val, steps_per_epoch, label_prefix="", color_t="tab:blue", color_v="tab:orange"):
    if train:
        ax.plot([s for s, _ in train], [l for _, (_, l) in train], color=color_t, lw=1.2, alpha=0.85,
                label=f"{label_prefix}training loss")
    if val:
        ax.plot([s for s, _ in val], [l for _, (_, l) in val], color=color_v, lw=2, marker="o", ms=5,
                label=f"{label_prefix}validation loss")
        best = min(val, key=lambda x: x[1][1])
        ax.scatter([best[0]], [best[1][1]], s=140, facecolors="none", edgecolors=color_v, lw=2, zorder=5)
    if steps_per_epoch:
        last = max([s for s, _ in train] + [s for s, _ in val] + [0])
        e = 1
        while e * steps_per_epoch <= last + 1e-6:
            ax.axvline(e * steps_per_epoch, color="grey", ls=":", lw=0.8)
            e += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/loss_comparison.png"), help="overlay plot (several runs)")
    a = ap.parse_args()

    data = []
    for run in a.runs:
        train, val, spe = read_losses(run)
        print_table(run.name, train, val)
        fig, ax = plt.subplots(figsize=(9, 5))
        draw(ax, train, val, spe)
        ax.set_title(f"{run.name}: training vs validation loss  (circle = best validation loss)")
        ax.set_xlabel("optimizer step  (dotted lines = epoch boundaries)")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png = run / "loss_plot.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"saved {png}")
        data.append((run.name, train, val, spe))

    if len(data) > 1:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, (name, train, val, spe) in enumerate(data):
            c = colors[i % len(colors)]
            if train:
                ax.plot([s for s, _ in train], [l for _, (_, l) in train], color=c, lw=1, alpha=0.5,
                        label=f"{name} train")
            if val:
                ax.plot([s for s, _ in val], [l for _, (_, l) in val], color=c, lw=2, marker="o", ms=4,
                        label=f"{name} validation")
        ax.set_title("Training (thin) vs validation (thick) loss")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.out, dpi=130)
        plt.close(fig)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
