"""Save the CSC-LISTA comparison in the style of the training plots."""
import argparse
import json
from itertools import count
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_DIR = Path(__file__).resolve().parent


def save_png(fig, requested):
    path = requested if requested.suffix else requested.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    separator = "_" if path.stem[-1:].isdigit() else ""
    for number in count(1):
        suffix = "" if number == 1 else f"{separator}{number}"
        candidate = path.with_stem(path.stem + suffix)
        try:
            stream = candidate.open("xb")
        except FileExistsError:
            continue
        except IsADirectoryError:
            continue
        try:
            with stream:
                fig.savefig(stream, format="png", dpi=200)
        except BaseException:
            candidate.unlink()
            raise
        return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=RESULT_DIR / "json" / "metrics2.json",
                        help="Pick a specific metrics JSON")
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "csc_lista_vs_snn.png")
    parser.add_argument("--smoke-test", action="store_true", help="Label a small verification run")
    args = parser.parse_args()
    print(f"Reading metrics: {args.metrics}")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))

    keys = ("original", "fast")
    accuracy = [100 * metrics["accuracy"][key] for key in keys]
    times = [metrics["forward_seconds"][key] for key in keys]
    labels = ("Original CSC-LISTA", "SNN-assisted")
    colors = ("tab:blue", "tab:orange")
    images = metrics["test_images"]
    device = metrics["device"].upper()
    skipped = metrics.get("skipped_updates_per_layer", metrics["original_updates"] - metrics["fast_updates"])
    layers = ", ".join(map(str, metrics.get("approximate_layers", [3])))

    with plt.style.context("default"):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), layout="constrained")
        title = "CIFAR-10 smoke test" if args.smoke_test else "CIFAR-10 comparison"
        fig.suptitle(f"{title} ({images:,} test images)")
        for axis, values, ylabel, title, number_format, limit in (
            (axes[0], accuracy, "Accuracy (%)", "Classification accuracy",
             "%.2f%%", max(100, max(accuracy) + 4)),
            (axes[1], times, f"Total time for {images:,} images (s)", f"{device} inference time",
             "%.1f s", max(times) * 1.15),
        ):
            bars = axis.bar(labels, values, color=colors, width=0.55)
            axis.bar_label(bars, fmt=number_format, padding=4)
            axis.set(title=title, ylabel=ylabel, ylim=(0, limit))
            axis.set_axisbelow(True)
            axis.grid(axis="y")

        note = f"Layer(s) {layers}: {skipped} LISTA update(s) replaced per layer"
        if args.smoke_test:
            note += f"  |  Fit: {metrics['fit_images']} images, {metrics['epochs']} epoch(s)"
        if device == "CPU":
            note += f"  |  {metrics['threads']} CPU threads"
        if "test_batch_size" in metrics:
            note += f"  |  Batch size {metrics['test_batch_size']}"
        layout = fig.get_layout_engine()
        layout.set(rect=(0, 0.055, 1, 0.945))
        fig.text(0.5, 0.025, note, ha="center", fontsize=9)
        try:
            output_path = save_png(fig, args.output)
        finally:
            plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
