"""Assemble the grid of panels for the data figures.

Three layouts:

* ``assemble_grid`` (3-column): Wigner heatmap, W(x,0), P_{π/2}(x,0).
  The legacy layout used by render_eigen.py and render_cat.py.

* ``assemble_grid_4col``: + the action-capacity kernel column.

* ``assemble_grid_5col``: + the rotation-averaged portrait. Reads as a
  full pipeline: Wigner portrait, slice, action-capacity kernel,
  matched cross-section, action-capacity portrait.

All heatmap columns use ``aspect='equal'`` so cell ellipses keep their
true shape. Cross-section panels have no aspect lock.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..state import State
from .panels import (
    P_theta_cross_section,
    matched_kernel_heatmap,
    tilde_W_heatmap,
    wigner_cross_section,
    wigner_heatmap,
)


# ---------------------------------------------------------------------------
# 3-column layout (legacy)
# ---------------------------------------------------------------------------

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
    """Build a 3-column figure as a clean grid with aspect-equal heatmaps."""
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

        wigner_heatmap(ax_h, state,
                       show_extended=show_extended, show_squeezed=show_squeezed)
        ax_h.set_aspect("equal", adjustable="box")

        wigner_cross_section(ax_w, state)
        P_theta_cross_section(ax_p, state)

        if i == 0:
            ax_h.set_title(column_titles[0])
            ax_w.set_title(column_titles[1])
            ax_p.set_title(column_titles[2])

        ax_h.set_ylabel(r"$p/p_0$")
        ax_w.set_ylabel(r"$W(x, 0)$")
        ax_p.set_ylabel(r"$P_{\pi/2}(x, 0)$")

        if i == n_rows - 1:
            ax_h.set_xlabel(r"$x/x_0$")
            ax_w.set_xlabel(r"$x/x_0$")
            ax_p.set_xlabel(r"$x/x_0$")

    return fig


# ---------------------------------------------------------------------------
# 4-column layout: + action-capacity-kernel heatmap column
# ---------------------------------------------------------------------------

_DEFAULT_4COL_TITLES = (
    "Wigner Function",
    "Wigner Cross-Section",
    r"Action-Capacity Kernel ($\theta = \pi/2$)",
    "Convolved Cross-Section",
)


def assemble_grid_4col(
    states: list[State],
    *,
    panel_width: float = 1.5,
    panel_height: float = 1.5,
    h_pad: float = 0.35,
    w_pad: float = 0.50,
    margin_top: float = 0.40,
    margin_bottom: float = 0.40,
    margin_left: float = 1.20,
    margin_right: float = 0.15,
    column_titles: tuple[str, str, str, str] = _DEFAULT_4COL_TITLES,
    row_labels: list[str] | None = None,
    show_extended: bool = True,
    show_squeezed: bool = True,
    theta: float | None = None,
) -> Figure:
    """Build a 4-column figure adding the action-capacity-kernel column."""
    import numpy as np
    if theta is None:
        theta = np.pi / 2

    n_rows = len(states)
    n_cols = 4

    if row_labels is not None and len(row_labels) != n_rows:
        raise ValueError(
            f"row_labels has {len(row_labels)} entries, expected {n_rows}"
        )

    fig_width = margin_left + n_cols * panel_width + (n_cols - 1) * w_pad + margin_right
    fig_height = (
        margin_top + n_rows * panel_height + (n_rows - 1) * h_pad + margin_bottom
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_cols,
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
        ax_k = fig.add_subplot(gs[i, 2])
        ax_p = fig.add_subplot(gs[i, 3])

        wigner_heatmap(ax_h, state,
                       show_extended=show_extended, show_squeezed=show_squeezed)
        ax_h.set_aspect("equal", adjustable="box")

        wigner_cross_section(ax_w, state)

        matched_kernel_heatmap(ax_k, state, theta=theta,
                               show_extended=show_extended,
                               show_squeezed=show_squeezed)
        ax_k.set_aspect("equal", adjustable="box")

        P_theta_cross_section(ax_p, state, theta=theta)

        if i == 0:
            ax_h.set_title(column_titles[0])
            ax_w.set_title(column_titles[1])
            ax_k.set_title(column_titles[2])
            ax_p.set_title(column_titles[3])

        ax_h.set_ylabel(r"$p/p_0$")
        ax_w.set_ylabel(r"$W(x, 0)$")
        ax_k.set_ylabel(r"$p/p_0$")
        ax_p.set_ylabel(r"$P_{\pi/2}(x, 0)$")

        if i == n_rows - 1:
            ax_h.set_xlabel(r"$x/x_0$")
            ax_w.set_xlabel(r"$x/x_0$")
            ax_k.set_xlabel(r"$x/x_0$")
            ax_p.set_xlabel(r"$x/x_0$")

        if row_labels is not None:
            pos_h = ax_h.get_position()
            y_center = 0.5 * (pos_h.y0 + pos_h.y1)
            x_label = pos_h.x0 - 0.8 * w_pad / fig_width
            fig.text(
                x_label, y_center,
                row_labels[i],
                rotation=90, ha="center", va="center",
                fontsize=plt.rcParams["axes.titlesize"],
            )

    return fig


# ---------------------------------------------------------------------------
# 5-column layout: + rotation-averaged portrait column
# ---------------------------------------------------------------------------

# Col 1 and col 5 are both 2D phase-space portraits of the state and
# share the noun "Portrait" so the pair reads in parallel. The
# Wigner-vs-action-capacity comparison is the figure's punchline; the
# titles state it directly.
_DEFAULT_5COL_TITLES = (
    "Wigner Portrait",
    "Wigner Cross-Section",
    r"Action-Capacity Kernel ($\theta = \pi/2$)",
    "Convolved Cross-Section",
    "Action-Capacity Portrait",
)


def assemble_grid_5col(
    states: list[State],
    *,
    panel_width: float = 1.5,
    panel_height: float = 1.5,
    h_pad: float = 0.35,
    w_pad: float = 0.70,
    margin_top: float = 0.40,
    margin_bottom: float = 0.40,
    margin_left: float = 1.20,
    margin_right: float = 0.15,
    column_titles: tuple[str, str, str, str, str] = _DEFAULT_5COL_TITLES,
    row_labels: list[str] | None = None,
    show_extended: bool = True,
    show_squeezed: bool = True,
    theta: float | None = None,
    tilde_W_n_theta: int = 360,
) -> Figure:
    """Build the 5-column numerical-results figure.

    Layout per row, left to right:
      Col 1: W(x, p) with extended + inscribed cell overlays.
      Col 2: W(x, 0).
      Col 3: K_{π/2}(x, p) with extended + inscribed cell overlays.
      Col 4: P_{π/2}(x, 0).
      Col 5: tilde_W(x, p) with extended + Zurek cell overlays.

    Cols 1, 3, 5 are aspect-equal phase-space heatmaps anchored at
    state.cell_center_x. Cols 2 and 4 are cross-sections.

    ``tilde_W_n_theta`` controls the angular sampling of the rotation
    average in col 5. 360 is the publication default; 60-90 is fast
    enough for layout iteration but shows visible angular striping at
    the panel corners.

    ``row_labels``: one short label per row, placed in the left margin
    rotated 90°.
    """
    import numpy as np
    if theta is None:
        theta = np.pi / 2

    n_rows = len(states)
    n_cols = 5

    if row_labels is not None and len(row_labels) != n_rows:
        raise ValueError(
            f"row_labels has {len(row_labels)} entries, expected {n_rows}"
        )

    fig_width = margin_left + n_cols * panel_width + (n_cols - 1) * w_pad + margin_right
    fig_height = (
        margin_top + n_rows * panel_height + (n_rows - 1) * h_pad + margin_bottom
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_cols,
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
        ax_k = fig.add_subplot(gs[i, 2])
        ax_p = fig.add_subplot(gs[i, 3])
        ax_t = fig.add_subplot(gs[i, 4])

        # Cols 1 and 3: extended + squeezed cells (no Zurek).
        wigner_heatmap(
            ax_h, state,
            show_extended=False, show_squeezed=False,
            show_zurek=False,
        )
        ax_h.set_aspect("equal", adjustable="box")

        wigner_cross_section(ax_w, state)

        matched_kernel_heatmap(
            ax_k, state, theta=theta,
            show_extended=show_extended, show_squeezed=show_squeezed,
            show_zurek=False,
        )
        ax_k.set_aspect("equal", adjustable="box")

        P_theta_cross_section(ax_p, state, theta=theta)

        # Col 5: extended + Zurek cells (no squeezed).
        tilde_W_heatmap(
            ax_t, state,
            n_theta=tilde_W_n_theta,
            show_extended=show_extended, show_squeezed=False,
            show_zurek=True,
        )
        ax_t.set_aspect("equal", adjustable="box")

        if i == 0:
            ax_h.set_title(column_titles[0])
            ax_w.set_title(column_titles[1])
            ax_k.set_title(column_titles[2])
            ax_p.set_title(column_titles[3])
            ax_t.set_title(column_titles[4])

        ax_h.set_ylabel(r"$p/p_0$")
        ax_w.set_ylabel(r"$W(x, 0)$")
        ax_k.set_ylabel(r"$p/p_0$")
        ax_p.set_ylabel(r"$P_{\pi/2}(x, 0)$")
        ax_t.set_ylabel(r"$p/p_0$")

        if i == n_rows - 1:
            for ax in (ax_h, ax_w, ax_k, ax_p, ax_t):
                ax.set_xlabel(r"$x/x_0$")

        if row_labels is not None:
            pos_h = ax_h.get_position()
            y_center = 0.5 * (pos_h.y0 + pos_h.y1)
            x_label = pos_h.x0 - 0.8 * w_pad / fig_width
            fig.text(
                x_label, y_center,
                row_labels[i],
                rotation=90, ha="center", va="center",
                fontsize=plt.rcParams["axes.titlesize"],
            )

    return fig


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_grid(fig: Figure, out_path: str | Path) -> None:
    """Save the grid to PDF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path}")
