"""The panels of each row of the data figures.

Column 1: W(x, p) heatmap with extended + squeezed cell overlay
          (diverging colormap).
Column 2: W(x, 0) cross-section — black line, red/blue fill.
Column 3: K_{π/2}(x, p) heatmap with extended + squeezed cell overlay
          (sequential — upper half of the diverging palette). Action-
          capacity kernel: Wigner function of the inscribed quantum
          blob at θ = π/2.
Column 4: P_{π/2}(x, 0) — non-negative, red fill.
Column 5: tilde_W(x, p) = (1/N_θ) Σ P_θ — rotation-averaged portrait,
          with extended + Zurek cell overlay (sequential). The inner
          ellipse is the Zurek sub-Planck cell with semi-axes (δx, δp);
          the polar dual of the extended cell, marking the inner
          resolution envelope of the rotation average.

Every drawn line in this figure — cell-overlay ellipses, cross-section
traces, zero baselines — uses ``overlays.LINEWIDTH``. One constant
governs all of them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm

from ..cells import ExtendedCell, cell_a_delta_x, squeezed_cell_at, zurek_cell
from ..convolve import convolve_W_with_K, cross_section_at_p0
from ..kernels import K_theta_mesh
from ..state import State
from .overlays import (
    LINEWIDTH,
    extended_cell_patch,
    squeezed_cell_patch,
    zurek_cell_patch,
)

# Colors drawn from the diverging colormap so the row reads as one palette.
_RDBU = plt.get_cmap("RdBu_r")
NEGATIVE_FILL = _RDBU(0.18)   # blue, for W < 0 in column 2
POSITIVE_FILL = _RDBU(0.82)   # red, for P_θ ≥ 0 in column 4


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _display_extent(state: State) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Return masked x and p slices plus matplotlib extent for the
    state's display window. Shared by all heatmap panels so cols 1, 3,
    5 render at identical coordinates."""
    assert state.x_int is not None and state.p_int is not None
    w = state.window
    ix_mask = (state.x_int >= w.x_min) & (state.x_int <= w.x_max)
    ip_mask = (state.p_int >= w.p_min) & (state.p_int <= w.p_max)
    x_clip = state.x_int[ix_mask]
    p_clip = state.p_int[ip_mask]
    extent = [x_clip[0], x_clip[-1], p_clip[0], p_clip[-1]]
    return ix_mask, ip_mask, extent


def _draw_cells(
    ax: Axes,
    state: State,
    *,
    show_extended: bool,
    show_squeezed: bool,
    show_zurek: bool,
    overlay_color: str,
    overlay_linewidth: float,
) -> None:
    """Overlay cells on a heatmap panel.

    `show_extended` draws the outer cell (semi-axes Δx, Δp).
    `show_squeezed` draws the inscribed cell at θ = π/2 (cols 1 & 3).
    `show_zurek` draws the Zurek sub-Planck cell (col 5).

    All cells are anchored at `state.cell_center_x`.
    """
    extended = ExtendedCell(
        Delta_x=state.rs.Delta_x,
        Delta_p=state.rs.Delta_p,
        center=(state.cell_center_x, 0.0),
    )
    if show_extended:
        ax.add_patch(extended_cell_patch(
            extended, edgecolor=overlay_color, linewidth=overlay_linewidth,
        ))
    if show_squeezed:
        sq = cell_a_delta_x(extended, hbar=state.hbar)
        ax.add_patch(squeezed_cell_patch(
            sq, edgecolor=overlay_color, linewidth=overlay_linewidth,
        ))
    if show_zurek:
        zk = zurek_cell(extended, hbar=state.hbar)
        ax.add_patch(zurek_cell_patch(
            zk, edgecolor=overlay_color, linewidth=overlay_linewidth,
        ))


def _apply_window(ax: Axes, state: State) -> None:
    """Apply the state's display window x/y limits and ticks. Shared
    across all panels so a row reads with a single x-axis."""
    w = state.window
    ax.set_xlim(w.x_min, w.x_max)
    ax.set_ylim(w.p_min, w.p_max)
    if w.x_ticks:
        ax.set_xticks(list(w.x_ticks))
    if w.p_ticks:
        ax.set_yticks(list(w.p_ticks))


# ---------------------------------------------------------------------------
# Column 1: W(x, p) heatmap
# ---------------------------------------------------------------------------

def wigner_heatmap(
    ax: Axes,
    state: State,
    *,
    show_extended: bool = False,
    show_squeezed: bool = False,
    show_zurek: bool = False,
    overlay_color: str = "black",
    overlay_linewidth: float = LINEWIDTH,
) -> None:
    """Render W(x, p) as a diverging-colormap heatmap with cell overlays."""
    assert state.W is not None
    ix_mask, ip_mask, extent = _display_extent(state)
    W_clip = state.W[np.ix_(ix_mask, ip_mask)]
    W_max = float(np.max(np.abs(W_clip)))
    if W_max == 0:
        W_max = 1.0

    ax.imshow(
        W_clip.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=_RDBU,
        norm=TwoSlopeNorm(vmin=-W_max, vcenter=0, vmax=W_max),
        interpolation="bilinear",
    )

    _draw_cells(
        ax, state,
        show_extended=show_extended,
        show_squeezed=show_squeezed,
        show_zurek=show_zurek,
        overlay_color=overlay_color,
        overlay_linewidth=overlay_linewidth,
    )
    _apply_window(ax, state)


# ---------------------------------------------------------------------------
# Column 2: W(x, 0) cross-section
# ---------------------------------------------------------------------------

def wigner_cross_section(
    ax: Axes,
    state: State,
    *,
    n_display: int = 500,
) -> np.ndarray:
    """Plot W(x, 0). Positive regions shaded red, negatives shaded blue,
    matching the heatmap colormap. Data trace and zero baseline are
    drawn at the shared LINEWIDTH so the cross-section panels read at
    the same line weight as the heatmap overlays."""
    assert state.W is not None and state.x_int is not None and state.p_int is not None
    w = state.window
    x_display = np.linspace(w.x_min, w.x_max, n_display)

    W_cross = cross_section_at_p0(state.W, state.x_int, state.p_int, x_display)

    ax.plot(x_display, W_cross, color="black", linewidth=LINEWIDTH)
    ax.fill_between(
        x_display, W_cross, 0,
        where=(W_cross > 0),
        color=POSITIVE_FILL, alpha=0.6, edgecolor="none", interpolate=True,
    )
    ax.fill_between(
        x_display, W_cross, 0,
        where=(W_cross < 0),
        color=NEGATIVE_FILL, alpha=0.6, edgecolor="none", interpolate=True,
    )
    W_lim = 1.1 * float(np.max(np.abs(W_cross)))
    ax.axhline(0, color="0.6", linewidth=LINEWIDTH, zorder=0)
    ax.set_xlim(w.x_min, w.x_max)
    ax.set_ylim(-W_lim, W_lim)
    if w.x_ticks:
        ax.set_xticks(list(w.x_ticks))
    return W_cross


# ---------------------------------------------------------------------------
# Column 3: action-capacity kernel K_{π/2}(x, p)
# ---------------------------------------------------------------------------

def matched_kernel_heatmap(
    ax: Axes,
    state: State,
    *,
    theta: float = np.pi / 2,
    show_extended: bool = True,
    show_squeezed: bool = True,
    show_zurek: bool = False,
    overlay_color: str = "black",
    overlay_linewidth: float = LINEWIDTH,
) -> None:
    """Render the action-capacity kernel K_θ(x, p) as a single-hue heatmap
    with cell overlays.

    The kernel is built at the same center as the cell overlays
    (``state.cell_center_x``), so the K-blob sits inside the inscribed
    squeezed cell drawn on top of it.

    Colormap: the upper half of the RdBu_r palette used in column 1, so
    the two heatmaps read as one color scheme — column 1 uses both
    sides of zero, column 3 uses only the red side because K_θ ≥ 0
    everywhere by construction.
    """
    assert state.x_int is not None and state.p_int is not None
    extended = ExtendedCell(
        Delta_x=state.rs.Delta_x,
        Delta_p=state.rs.Delta_p,
        center=(state.cell_center_x, 0.0),
    )
    cell = squeezed_cell_at(theta, extended, hbar=state.hbar)
    xx, pp = np.meshgrid(state.x_int, state.p_int, indexing="ij")
    K = K_theta_mesh(cell, xx, pp, hbar=state.hbar)

    ix_mask, ip_mask, extent = _display_extent(state)
    K_clip = K[np.ix_(ix_mask, ip_mask)]
    K_max = float(K_clip.max())
    if K_max == 0:
        K_max = 1.0

    ax.imshow(
        K_clip.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=_RDBU,
        norm=Normalize(vmin=-K_max, vmax=K_max),
        interpolation="bilinear",
    )

    _draw_cells(
        ax, state,
        show_extended=show_extended,
        show_squeezed=show_squeezed,
        show_zurek=show_zurek,
        overlay_color=overlay_color,
        overlay_linewidth=overlay_linewidth,
    )
    _apply_window(ax, state)


# ---------------------------------------------------------------------------
# Column 4: P_{π/2}(x, 0) cross-section
# ---------------------------------------------------------------------------

def P_theta_cross_section(
    ax: Axes,
    state: State,
    *,
    theta: float = np.pi / 2,
    n_display: int = 500,
) -> np.ndarray:
    """Plot P_θ(x, 0) over the row's display window. Defaults to θ = π/2.
    Data trace drawn at the shared LINEWIDTH."""
    assert state.W is not None and state.x_int is not None and state.p_int is not None

    # Kernel center at the integration grid midpoint, where
    # fftconvolve(mode='same') aligns the kernel's array center with W's
    # array center.
    x_mid = 0.5 * (state.x_int[0] + state.x_int[-1])
    p_mid = 0.5 * (state.p_int[0] + state.p_int[-1])

    extended = ExtendedCell(
        Delta_x=state.rs.Delta_x,
        Delta_p=state.rs.Delta_p,
        center=(x_mid, p_mid),
    )
    cell = squeezed_cell_at(theta, extended, hbar=state.hbar)
    xx, pp = np.meshgrid(state.x_int, state.p_int, indexing="ij")
    K = K_theta_mesh(cell, xx, pp, hbar=state.hbar)

    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])
    P = convolve_W_with_K(state.W, K, dx, dp)

    # Hudson sanity check.
    P_min = float(P.min())
    if P_min < -1e-6:
        import warnings
        warnings.warn(
            f"State {state.name!r}: P_θ has minimum {P_min:.2e}; "
            "Hudson's theorem should give P ≥ 0. Likely numerical."
        )

    w = state.window
    x_display = np.linspace(w.x_min, w.x_max, n_display)
    P_cross = cross_section_at_p0(P, state.x_int, state.p_int, x_display)

    ax.plot(x_display, P_cross, color="black", linewidth=LINEWIDTH)
    ax.fill_between(
        x_display, np.maximum(P_cross, 0), 0,
        color=POSITIVE_FILL, alpha=0.6, edgecolor="none",
    )
    P_lim = 1.1 * float(P_cross.max()) if P_cross.max() > 0 else 1.0
    ax.set_xlim(w.x_min, w.x_max)
    ax.set_ylim(0, P_lim)
    if w.x_ticks:
        ax.set_xticks(list(w.x_ticks))
    return P_cross


# ---------------------------------------------------------------------------
# Column 5: tilde_W(x, p) rotation-averaged portrait
# ---------------------------------------------------------------------------

def tilde_W_heatmap(
    ax: Axes,
    state: State,
    *,
    n_theta: int = 360,
    show_extended: bool = True,
    show_squeezed: bool = False,
    show_zurek: bool = True,
    overlay_color: str = "black",
    overlay_linewidth: float = LINEWIDTH,
) -> np.ndarray:
    """Render the rotation-averaged portrait tilde_W(x, p) = (1/N_θ) Σ P_θ.

    The inscribed-family parameter θ is integrated out by directly
    averaging the 2D convolutions P_θ = W * K_θ at evenly spaced θ in
    [0, π). Non-negative by construction: each P_θ ≥ 0 by Hudson, and
    a sum of non-negative functions is non-negative.

    The kernel at each θ is centered at the integration-grid midpoint
    so fftconvolve(mode='same') aligns it with W. ``n_theta`` controls
    the angular sampling; 360 is comfortably above the rule-of-thumb
    threshold (π r_max / δ_perp) for every state in the manuscript's
    library, so no angular striping is visible in the published figure.

    Colormap: upper half of the RdBu_r palette, matching column 3.
    Cell overlay: extended cell (outer) plus Zurek cell (inner). The
    Zurek cell marks the inner resolution envelope of the rotation
    average — the smallest scale on which tilde_W carries structure.

    Returns the full 2D tilde_W array (on state.x_int × state.p_int)
    for downstream use.
    """
    assert state.W is not None and state.x_int is not None and state.p_int is not None

    # Kernel center at integration grid midpoint for fftconvolve alignment.
    x_mid = 0.5 * (state.x_int[0] + state.x_int[-1])
    p_mid = 0.5 * (state.p_int[0] + state.p_int[-1])
    extended_for_kernel = ExtendedCell(
        Delta_x=state.rs.Delta_x,
        Delta_p=state.rs.Delta_p,
        center=(x_mid, p_mid),
    )

    xx, pp = np.meshgrid(state.x_int, state.p_int, indexing="ij")
    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])

    # Direct sum of P_θ over θ ∈ [0, π) at evenly spaced angles. The
    # endpoint π is excluded because K_θ has Z_2 symmetry (K_θ = K_{θ+π}),
    # so including it would double-count.
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    tilde_W = np.zeros_like(state.W)
    for theta in thetas:
        cell = squeezed_cell_at(theta, extended_for_kernel, hbar=state.hbar)
        K = K_theta_mesh(cell, xx, pp, hbar=state.hbar)
        P_theta = convolve_W_with_K(state.W, K, dx, dp)
        tilde_W += P_theta
    tilde_W /= n_theta

    # Hudson sanity check: tilde_W is a non-negative sum of non-negatives.
    tW_min = float(tilde_W.min())
    if tW_min < -1e-6:
        import warnings
        warnings.warn(
            f"State {state.name!r}: tilde_W has minimum {tW_min:.2e}; "
            "should be ≥ 0 to floating-point precision. Likely numerical."
        )

    # Clip to display window for plotting.
    ix_mask, ip_mask, extent = _display_extent(state)
    tW_clip = tilde_W[np.ix_(ix_mask, ip_mask)]
    tW_max = float(tW_clip.max())
    if tW_max == 0:
        tW_max = 1.0

    ax.imshow(
        tW_clip.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=_RDBU,
        norm=Normalize(vmin=-tW_max, vmax=tW_max),
        interpolation="bilinear",
    )

    _draw_cells(
        ax, state,
        show_extended=show_extended,
        show_squeezed=show_squeezed,
        show_zurek=show_zurek,
        overlay_color=overlay_color,
        overlay_linewidth=overlay_linewidth,
    )
    _apply_window(ax, state)

    return tilde_W
