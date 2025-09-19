import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def import_large_schwarz_class() -> object:
    """Import and return the LargeSchwarz class from helpers/large_envs.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Ensure parent directory (sim_env/baselines) is importable as a package root
    baselines_dir = script_dir
    parent_dir = os.path.dirname(baselines_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    # Now we can import helpers.large_envs as a module
    from helpers.large_envs import LargeSchwarz  # type: ignore
    return LargeSchwarz


def build_color_grid(occupancy: np.ndarray, semantic_grid: list[list[dict]]) -> np.ndarray:
    """Create an RGB image from occupancy and semantic grids.

    - occupancy == 1 -> black
    - occupancy == 0 -> white
    - cells with a 'sign' -> blue (override)
    """
    rows, cols = occupancy.shape
    rgb = np.ones((rows, cols, 3), dtype=float)

    # Walls (1) to black
    wall_mask = occupancy == 1
    rgb[wall_mask] = np.array([0.0, 0.0, 0.0])

    # Highlight sign cells in blue
    for y in range(rows):
        for x in range(cols):
            cell_meta = semantic_grid[y][x]
            if isinstance(cell_meta, dict) and ("sign" in cell_meta):
                rgb[y, x] = np.array([0.0, 0.45, 1.0])

    return rgb


def draw_grid(ax: plt.Axes, occupancy: np.ndarray, semantic_grid: list[list[dict]]):
    rgb = build_color_grid(occupancy, semantic_grid)
    rows, cols = occupancy.shape

    ax.imshow(rgb, origin="upper", interpolation="nearest")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.6)

    # Remove major ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Overlay room numbers wherever present
    for y in range(rows):
        for x in range(cols):
            cell_meta = semantic_grid[y][x]
            if isinstance(cell_meta, dict) and ("room_number" in cell_meta):
                room = cell_meta.get("room_number")
                if room is not None:
                    ax.text(
                        x,
                        y,
                        str(room),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                        fontweight="bold",
                    )


def main():
    parser = argparse.ArgumentParser(description="Visualize LargeSchwarz environment grid.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save image (PNG). Defaults to test_imgs/large_schwarz_grid.png",
    )
    args = parser.parse_args()

    LargeSchwarz = import_large_schwarz_class()
    env = LargeSchwarz()
    occupancy = env.occupancy_grid
    semantic_grid = env.semantic_grid

    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "test_imgs", "large_schwarz_grid.png")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rows, cols = occupancy.shape
    scale = 0.12  # large map; keep image reasonable
    fig_w = max(6.0, cols * scale)
    fig_h = max(6.0, rows * scale)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    draw_grid(ax, occupancy, semantic_grid)
    plt.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()


