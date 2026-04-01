"""
Placement Visualizer
====================

Matplotlib-based visualization tools for examining placement results.

Usage (standalone):
    python visualize.py

Usage (from code):
    from visualize import plot_placement, plot_loss_history
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from placement import (
    generate_placement_input,
    train_placement,
    calculate_cells_with_overlaps,
    CellFeatureIdx,
)


# ── colours ────────────────────────────────────────────────────────────────────
MACRO_FACE   = "#f4a261"   # orange
MACRO_EDGE   = "#e76f51"
STD_FACE     = "#a8dadc"   # teal-blue
STD_EDGE     = "#457b9d"
OVERLAP_FACE = "#e63946"   # red (drawn on top of normal colour)
OVERLAP_ALPHA = 0.45
WIRE_COLOR   = "#6c757d"
WIRE_ALPHA   = 0.25


# ── helpers ────────────────────────────────────────────────────────────────────

def _num_macros(cell_features: torch.Tensor) -> int:
    """Infer the number of macros by counting cells with height > 1."""
    heights = cell_features[:, CellFeatureIdx.HEIGHT].detach()
    return int((heights > 1.0).sum().item())


def _draw_cells(ax, cell_features: torch.Tensor, cells_with_overlaps: set):
    """Draw all cells onto *ax*, colour-coded by type, overlaps highlighted."""
    n_macros = _num_macros(cell_features)
    positions = cell_features[:, 2:4].detach().numpy()
    widths    = cell_features[:, CellFeatureIdx.WIDTH].detach().numpy()
    heights   = cell_features[:, CellFeatureIdx.HEIGHT].detach().numpy()
    N         = cell_features.shape[0]

    for i in range(N):
        x = positions[i, 0] - widths[i] / 2
        y = positions[i, 1] - heights[i] / 2
        is_macro   = i < n_macros
        is_overlap = i in cells_with_overlaps

        face  = MACRO_FACE if is_macro else STD_FACE
        edge  = MACRO_EDGE if is_macro else STD_EDGE
        lw    = 0.8

        ax.add_patch(Rectangle(
            (x, y), widths[i], heights[i],
            facecolor=face, edgecolor=edge, linewidth=lw, alpha=0.85, zorder=2,
        ))

        if is_overlap:
            ax.add_patch(Rectangle(
                (x, y), widths[i], heights[i],
                facecolor=OVERLAP_FACE, edgecolor=OVERLAP_FACE,
                linewidth=1.5, alpha=OVERLAP_ALPHA, zorder=3,
            ))


def _draw_wires(ax, cell_features: torch.Tensor, pin_features: torch.Tensor,
                edge_list: torch.Tensor):
    """Draw wire connections as thin lines between connected pin centres."""
    if edge_list.shape[0] == 0:
        return

    cell_pos   = cell_features[:, 2:4].detach()
    cell_idx   = pin_features[:, 0].long()
    pin_abs_x  = cell_pos[cell_idx, 0] + pin_features[:, 1]
    pin_abs_y  = cell_pos[cell_idx, 1] + pin_features[:, 2]

    srcs = edge_list[:, 0].long()
    tgts = edge_list[:, 1].long()

    xs = pin_abs_x.detach().numpy()
    ys = pin_abs_y.detach().numpy()

    for s, t in zip(srcs.numpy(), tgts.numpy()):
        ax.plot([xs[s], xs[t]], [ys[s], ys[t]],
                color=WIRE_COLOR, alpha=WIRE_ALPHA, linewidth=0.4, zorder=1)


def _set_limits(ax, cell_features: torch.Tensor, margin_frac: float = 0.05):
    positions = cell_features[:, 2:4].detach().numpy()
    widths    = cell_features[:, CellFeatureIdx.WIDTH].detach().numpy()
    heights   = cell_features[:, CellFeatureIdx.HEIGHT].detach().numpy()

    x_min = (positions[:, 0] - widths  / 2).min()
    x_max = (positions[:, 0] + widths  / 2).max()
    y_min = (positions[:, 1] - heights / 2).min()
    y_max = (positions[:, 1] + heights / 2).max()

    span   = max(x_max - x_min, y_max - y_min)
    margin = span * margin_frac
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)


# ── public API ─────────────────────────────────────────────────────────────────

def plot_placement(
    initial_cell_features: torch.Tensor,
    final_cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
    show_wires: bool = False,
    filename: str = "placement_result.png",
    show: bool = False,
):
    """Side-by-side initial vs final placement.

    Cells are colour-coded (macros = orange, std cells = teal).
    Cells involved in overlaps are highlighted in red.
    Pass show_wires=True to draw wire connections (can be noisy).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#f8f9fa")

    panels = [
        (ax1, initial_cell_features, "Initial placement"),
        (ax2, final_cell_features,   "Final placement"),
    ]

    for ax, cf, title in panels:
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linewidth=0.5)

        overlapping = calculate_cells_with_overlaps(cf)
        n_macros    = _num_macros(cf)
        N           = cf.shape[0]

        if show_wires:
            _draw_wires(ax, cf, pin_features, edge_list)

        _draw_cells(ax, cf, overlapping)
        _set_limits(ax, cf)

        ax.set_title(
            f"{title}\n"
            f"{N} cells ({n_macros} macros, {N - n_macros} std)  |  "
            f"{len(overlapping)} overlapping",
            fontsize=11, pad=8,
        )
        ax.set_xlabel("x (units)")
        ax.set_ylabel("y (units)")

    # legend
    legend_handles = [
        mpatches.Patch(facecolor=MACRO_FACE, edgecolor=MACRO_EDGE, label="Macro"),
        mpatches.Patch(facecolor=STD_FACE,   edgecolor=STD_EDGE,   label="Std cell"),
        mpatches.Patch(facecolor=OVERLAP_FACE, alpha=OVERLAP_ALPHA, label="Overlapping"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")
    if show:
        plt.show()
    plt.close()


def plot_loss_history(
    loss_history: dict,
    filename: str = "loss_history.png",
    show: bool = False,
):
    """Plot total, wirelength, and overlap loss over training epochs."""
    total_loss = loss_history["total_loss"]
    wl_loss    = loss_history["wirelength_loss"]
    ov_loss    = loss_history["overlap_loss"]

    epochs = range(len(total_loss))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("Training loss history", fontsize=13, y=1.01)

    specs = [
        (axes[0], total_loss, "Total loss",      "#4361ee"),
        (axes[1], wl_loss,    "Wirelength loss",  "#3a86ff"),
        (axes[2], ov_loss,    "Overlap loss",     "#e63946"),
    ]

    for ax, values, title, colour in specs:
        ax.set_facecolor("#ffffff")
        ax.plot(epochs, values, color=colour, linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_yscale("log")  # log scale shows convergence clearly

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")
    if show:
        plt.show()
    plt.close()


# ── standalone demo ────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    num_macros    = 3
    num_std_cells = 50

    print("Generating placement problem ...")
    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    total_cells  = cell_features.shape[0]
    total_area   = cell_features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii  = torch.rand(total_cells) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    print("Running optimizer ...")
    result = train_placement(cell_features, pin_features, edge_list, verbose=False)

    plot_placement(
        result["initial_cell_features"],
        result["final_cell_features"],
        pin_features,
        edge_list,
        show_wires=False,
        filename="placement_result.png",
    )

    plot_loss_history(
        result["loss_history"],
        filename="loss_history.png",
    )


if __name__ == "__main__":
    main()
