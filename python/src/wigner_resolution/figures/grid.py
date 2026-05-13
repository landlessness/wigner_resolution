"""Assemble the 4×3 grid of panels for the data figures.

Layout: a clean 4×3 grid with fixed panel dimensions. The Wigner heatmap
(column 1) has aspect='equal' so cell ellipses keep their true shape;
for anisotropic states (Δp ≫ Δx, or vice versa) matplotlib pads the
heatmap with whitespace inside the panel rather than distorting the
geometry. This is the visual mechanism by which the reader sees aspect
ratio directly.

Columns 2 and 3 (cross-sections) have no aspect lock.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..state import State
from .panels import (
    P_theta_cross_section,
    wigner_cross_section,
    wigner_heatmap,
)


def assemble_grid(
    states: list[State],
    *,
    panel_width: float = 1.7,
    panel_height: float = 1.5,
    h_pad: float = 0.35,
    w_pad: float = 0.55,
    margin_top: float = 0.40,
    margin_bottom: float = 0.40,
    margin_left: float = 0.55,
    margin_right: float = 0.15,
    column_titles: tuple[str, str, str] = (
        "Wigner Phase-Space",
        "Wigner Cross-Section",
        r"Convolved Cross-Section",
    ),
    show_extended: bool = True,
    show_squeezed: bool = True,
) -> Figure:
    """Build the 4×3 figure as a clean grid with aspect-equal heatmaps."""
    n_rows = len(states)

    fig_width = margin_left + 3 * panel_width + 2 * w_pad + margin_right
    fig_height = (
        margin_top + n_rows * panel_height + (n_rows - 1) * h_pad + margin_bottom
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=n_rows,
        ncols=3,
        figure=fig,
        left=margin_left / fig_width,
        right=1 - margin_right / fig_width,
        top=1 - margin_top / fig_height,
        bottom=margin_bottom / fig_height,
        hspace=h_pad / panel_height,
        wspace=w_pad / panel_width,
    )

    for i, state in enumerate(states):
        ax_h = fig.add_subplot(gs[i, 0])
        ax_w = fig.add_subplot(gs[i, 1])
        ax_p = fig.add_subplot(gs[i, 2])

        wigner_heatmap(ax_h, state, show_extended=show_extended,
                       show_squeezed=show_squeezed)
        ax_h.set_aspect("equal", adjustable="box")

        wigner_cross_section(ax_w, state)
        P_theta_cross_section(ax_p, state)

        if i == 0:
            ax_h.set_title(column_titles[0])
            ax_w.set_title(column_titles[1])
            ax_p.set_title(column_titles[2])

        # Y-labels appear on every row so cross-section panels are
        # unambiguous. The phase-space panel gets its label too.
        ax_h.set_ylabel(r"$p/p_0$")
        ax_w.set_ylabel(r"$W(x, 0)$")
        ax_p.set_ylabel(r"$P_{\pi/2}(x, 0)$")

        if i == n_rows - 1:
            ax_h.set_xlabel(r"$x/x_0$")
            ax_w.set_xlabel(r"$x/x_0$")
            ax_p.set_xlabel(r"$x/x_0$")

    return fig


def save_grid(fig: Figure, out_path: str | Path) -> None:
    """Save the grid to PDF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path}")
