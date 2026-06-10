from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINED_PROBES_ROOT = EXPERIMENT_ROOT / "outputs/trained_probes"
DEFAULT_OUT_PATH = DEFAULT_TRAINED_PROBES_ROOT / "sampson_error_by_layer.png"
LAYER_DIR_RE = re.compile(r"layer_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best pixel-space test Sampson error for each trained layer probe."
    )
    parser.add_argument(
        "--trained-probes-root",
        type=Path,
        default=DEFAULT_TRAINED_PROBES_ROOT,
        help="Directory containing layer_n subdirectories with metrics.csv files.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="PNG path for the generated plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )
    return parser.parse_args()


def layer_index(path: Path) -> int | None:
    match = LAYER_DIR_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def best_test_loss(metrics_path: Path) -> float:
    with metrics_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    losses: list[float] = []
    for row in rows:
        raw_loss = row.get("test_loss")
        if raw_loss is None or raw_loss == "":
            continue
        losses.append(float(raw_loss))

    if not losses:
        raise ValueError(f"No valid test_loss values found in {metrics_path}")

    return min(losses)


def load_layer_errors(trained_probes_root: Path) -> tuple[list[int], list[float]]:
    layer_errors: list[tuple[int, float]] = []
    for layer_dir in trained_probes_root.iterdir():
        if not layer_dir.is_dir():
            continue

        idx = layer_index(layer_dir)
        if idx is None:
            continue

        metrics_path = layer_dir / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file for layer {idx}: {metrics_path}")

        min_test_loss = best_test_loss(metrics_path)
        if min_test_loss < 0.0:
            raise ValueError(f"Negative test_loss found for layer {idx}: {min_test_loss}")
        layer_errors.append((idx, math.sqrt(min_test_loss)))

    if not layer_errors:
        raise ValueError(f"No layer_n metrics found under {trained_probes_root}")

    layer_errors.sort(key=lambda item: item[0])
    layers = [item[0] for item in layer_errors]
    errors = [item[1] for item in layer_errors]
    return layers, errors


def plot_layer_errors(layers: list[int], errors: list[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, errors, color="blue", marker="o", label="Best test loss")
    ax.set_xlabel("layer")
    ax.set_ylabel("Sampson Error (px)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> None:
    args = parse_args()
    trained_probes_root = args.trained_probes_root.resolve()
    out_path = args.out_path.resolve()

    layers, errors = load_layer_errors(trained_probes_root)
    plot_layer_errors(layers, errors, out_path)
    print(f"Wrote plot to {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
