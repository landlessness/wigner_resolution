"""The three panels of each row of the data figures.

Column 1: Wigner heatmap with cell overlay (W(x, p) on a 2D phase-space view).
Column 2: Wigner cross-section at p = 0 — black line, negative regions
          shaded in the blue tail of RdBu_r so the diverging-color
          encoding is consistent across the row.
Column 3: P_{π/2}(x, 0) — strictly non-negative by Hudson, shaded in the
          red tail of the same colormap so the cross-row eye finds
          "everything red" in column 3 versus "blue dips between red
          peaks" in column 2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm

from ..cells import ExtendedCell, cell_a_delta_x
from ..convolve import convolve_W_with_K, cross_section_at_p0
from ..kernels import K_theta_mesh
from ..state import State
from .overlays import extended_cell_patch, squeezed_cell_patch

# Colors drawn from the diverging colormap so the row reads as one palette.
_RDBU = plt.get_cmap("RdBu_r")
NEGATIVE_FILL = _RDBU(0.18)   # blue, for W < 0 in column 2
POSITIVE_FILL = _RDBU(0.82)   # red, for P_θ ≥ 0 in column 3


def wigner_heatmap(
    ax: Axes,
    state: State,
    *,
    show_extended: bool = True,
    show_squeezed: bool = True,
    overlay_color: str = "black",
    overlay_linewidth: float = 0.6,
) -> None:
    """Render W(x, p) as a heatmap with cell overlays."""
    assert state.W is not None and state.x_int is not None and state.p_int is not None

    w = state.window
    ix_mask = (state.x_int >= w.x_min) & (state.x_int <= w.x_max)
    ip_mask = (state.p_int >= w.p_min) & (state.p_int <= w.p_max)
    W_clip = state.W[np.ix_(ix_mask, ip_mask)]
    W_max = float(np.max(np.abs(W_clip)))
    if W_max == 0:
        W_max = 1.0

    extent = [
        state.x_int[ix_mask][0],
        state.x_int[ix_mask][-1],
        state.p_int[ip_mask][0],
        state.p_int[ip_mask][-1],
    ]
    ax.imshow(
        W_clip.T,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=_RDBU,
        norm=TwoSlopeNorm(vmin=-W_max, vcenter=0, vmax=W_max),
        interpolation="bilinear",
    )

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

    ax.set_xlim(w.x_min, w.x_max)
    ax.set_ylim(w.p_min, w.p_max)
    if w.x_ticks:
        ax.set_xticks(list(w.x_ticks))
    if w.p_ticks:
        ax.set_yticks(list(w.p_ticks))


def wigner_cross_section(
    ax: Axes,
    state: State,
    *,
    n_display: int = 500,
) -> np.ndarray:
    """Plot W(x, 0) over the row's display window.

    Positive regions are shaded red, negative regions blue, both drawn
    from the same RdBu_r colormap as the heatmap. The two-color fill
    encodes the sign of W directly and matches column 3's red shading
    (where everything is non-negative).

    Returns the cross-section array (for downstream peak counting).
    """
    assert state.W is not None and state.x_int is not None and state.p_int is not None
    w = state.window
    x_display = np.linspace(w.x_min, w.x_max, n_display)

    W_cross = cross_section_at_p0(state.W, state.x_int, state.p_int, x_display)

    ax.plot(x_display, W_cross, color="black", linewidth=0.8)
    ax.fill_between(
        x_display, W_cross, 0,
        where=(W_cross > 0),
        color=POSITIVE_FILL,
        alpha=0.6,
        edgecolor="none",
        interpolate=True,
    )
    ax.fill_between(
        x_display, W_cross, 0,
        where=(W_cross < 0),
        color=NEGATIVE_FILL,
        alpha=0.6,
        edgecolor="none",
        interpolate=True,
    )
    W_lim = 1.1 * float(np.max(np.abs(W_cross)))
    ax.axhline(0, color="0.6", linewidth=0.4, zorder=0)
    ax.set_xlim(w.x_min, w.x_max)
    ax.set_ylim(-W_lim, W_lim)
    if w.x_ticks:
        ax.set_xticks(list(w.x_ticks))
    return W_cross


def P_theta_cross_section(
    ax: Axes,
    state: State,
    *,
    theta: float = np.pi / 2,
    n_display: int = 500,
) -> np.ndarray:
    """Plot P_θ(x, 0) over the row's display window. Defaults to θ = π/2."""
    assert state.W is not None and state.x_int is not None and state.p_int is not None

    # Build the kernel centered at the integration grid's array midpoint.
    # fftconvolve(mode='same') aligns the kernel's array center with the
    # input array center, so positioning the kernel peak at the array
    # midpoint preserves x-features under convolution.
    x_mid = 0.5 * (state.x_int[0] + state.x_int[-1])
    p_mid = 0.5 * (state.p_int[0] + state.p_int[-1])

    extended = ExtendedCell(
        Delta_x=state.rs.Delta_x,
        Delta_p=state.rs.Delta_p,
        center=(x_mid, p_mid),
    )
    from ..cells import squeezed_cell_at
    cell = squeezed_cell_at(theta, extended, hbar=state.hbar)
    xx, pp = np.meshgrid(state.x_int, state.p_int, indexing="ij")
    K = K_theta_mesh(cell, xx, pp)

    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])
    P = convolve_W_with_K(state.W, K, dx, dp)

    # Hudson sanity check: tiny negative values from FFT roundoff are fine.
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

    ax.plot(x_display, P_cross, color="black", linewidth=0.8)
    ax.fill_between(
        x_display, np.maximum(P_cross, 0), 0,
        color=POSITIVE_FILL,
        alpha=0.6,
        edgecolor="none",
    )
    P_lim = 1.1 * float(P_cross.max()) if P_cross.max() > 0 else 1.0
    ax.set_xlim(w.x_min, w.x_max)
    ax.set_ylim(0, P_lim)
    if w.x_ticks:
        ax.set_xticks(list(w.x_ticks))
    return P_cross
