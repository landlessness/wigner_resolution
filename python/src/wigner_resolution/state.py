"""Per-state bundle and the pipelines that build it.

A ``State`` carries the wavefunction (or its number-basis Qobj), its RS
geometry, and an evaluated Wigner function on a uniform integration grid.
Downstream consumers (the figure code, the convolution code) read fields
off this bundle.

Two entry points cover all states in the manuscript:

* ``build_state_from_qobj`` — number-basis representation via QuTiP.
  Used for squeezed vacuum, harmonic oscillator eigenstates, and (when
  built) cat states. Wigner via ``qutip.wigner``.

* ``build_state_from_psi`` — position-basis sampled wavefunction. Used
  for eigenstates of arbitrary potentials (Morse, double-well) where the
  FD solver delivers ψ(x) on a grid. Wigner via ``wigner_from_psi`` (a
  thin wrapper around ``numpy.fft.fft``).

Both paths share a common downstream pipeline: cell-overlay anchoring,
display-window sizing, and tick selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

from .quantum import RSGeometry, numerical_covariance
from .wigner import wigner_from_psi, wigner_from_qobj, wigner_norm


@dataclass(frozen=True)
class DisplayWindow:
    """Phase-space display window for a single panel row.

    ``x_lim`` and ``p_lim`` are *half-widths* relative to the window's
    center. The window covers [x_center - x_lim, x_center + x_lim] in x
    and [p_center - p_lim, p_center + p_lim] in p.
    """

    x_lim: float
    p_lim: float
    x_center: float = 0.0
    p_center: float = 0.0
    x_ticks: tuple[float, ...] = ()
    p_ticks: tuple[float, ...] = ()

    @property
    def x_min(self) -> float:
        return self.x_center - self.x_lim

    @property
    def x_max(self) -> float:
        return self.x_center + self.x_lim

    @property
    def p_min(self) -> float:
        return self.p_center - self.p_lim

    @property
    def p_max(self) -> float:
        return self.p_center + self.p_lim


@dataclass
class State:
    """A fully-built phase-space state ready for plotting."""

    name: str
    rs: RSGeometry
    hbar: float
    window: DisplayWindow

    # Cell-overlay anchor. ``build_state_*`` sets this to the location of
    # max |W(x, 0)|, where the cell's resolution argument is visually most
    # legible. Systems can override.
    cell_center_x: float = 0.0

    # Populated by the build pipeline:
    W: np.ndarray | None = None
    x_int: np.ndarray | None = None
    p_int: np.ndarray | None = None


# ---------------------------------------------------------------------------
# QuTiP path: state given as a number-basis Qobj
# ---------------------------------------------------------------------------

def build_state_from_qobj(
    name: str,
    qobj: qt.Qobj,
    window: DisplayWindow,
    *,
    hbar: float = 1.0,
    n_x_int: int = 401,
    n_p_int: int = 401,
    x_pad_factor: float = 2.5,
    p_pad: float = 2.0,
    window_margin: float = 0.40,
    cell_center_x: float | None = None,
) -> State:
    """Build a State from a number-basis Qobj (ket or density matrix).

    Computes RS covariance via ``qutip.expect`` and the Wigner function
    via ``qutip.wigner``.
    """
    rs = _rs_geometry_from_qobj(qobj, hbar=hbar)
    x_int, p_int = _make_integration_grids(
        rs, n_x_int, n_p_int, x_pad_factor, p_pad, window_margin,
    )
    W = wigner_from_qobj(qobj, x_int, p_int, hbar=hbar)
    _check_wigner_norm(name, W, x_int, p_int)
    return _finalize_state(
        name=name, rs=rs, W=W, x_int=x_int, p_int=p_int,
        window=window, hbar=hbar,
        window_margin=window_margin, cell_center_x=cell_center_x,
    )


def _rs_geometry_from_qobj(qobj: qt.Qobj, *, hbar: float) -> RSGeometry:
    """RS geometry of a Qobj via expectation values of x, x², p, p².

    Uses QuTiP's position and momentum operators ``qt.position`` and
    ``qt.momentum``. The default scaling matches our hbar=1 convention.
    """
    if hbar != 1.0:
        raise NotImplementedError("Only ℏ=1 is supported via the QuTiP path.")
    N = qobj.dims[0][0]
    x_op = qt.position(N)
    p_op = qt.momentum(N)
    x_mean = float(qt.expect(x_op, qobj))
    x2_mean = float(qt.expect(x_op * x_op, qobj))
    p_mean = float(qt.expect(p_op, qobj))
    p2_mean = float(qt.expect(p_op * p_op, qobj))
    sigma_xx = x2_mean - x_mean ** 2
    sigma_pp = p2_mean - p_mean ** 2
    return RSGeometry(
        Delta_x=float(np.sqrt(2 * sigma_xx)),
        Delta_p=float(np.sqrt(2 * sigma_pp)),
        x_mean=x_mean,
        hbar=hbar,
    )


# ---------------------------------------------------------------------------
# Position-basis path: state given as a sampled ψ(x)
# ---------------------------------------------------------------------------

def build_state_from_psi(
    name: str,
    psi: np.ndarray,
    x_grid_psi: np.ndarray,
    window: DisplayWindow,
    *,
    hbar: float = 1.0,
    n_x_int: int = 401,
    n_p_int: int = 401,
    x_pad_factor: float = 2.5,
    p_pad: float = 2.0,
    window_margin: float = 0.40,
    cell_center_x: float | None = None,
) -> State:
    """Build a State from a sampled wavefunction ψ(x).

    Used for eigenstates of arbitrary potentials where ψ comes from a
    position-basis FD solver. Wigner via FFT of the autocorrelation
    (Leonhardt 1997, Ch. 5).
    """
    from scipy.interpolate import interp1d

    rs = numerical_covariance(psi, x_grid_psi, hbar=hbar)
    x_int, p_int = _make_integration_grids(
        rs, n_x_int, n_p_int, x_pad_factor, p_pad, window_margin,
    )

    # Interpolate ψ onto the integration grid, zero outside support.
    if np.iscomplexobj(psi):
        psi_re = interp1d(x_grid_psi, psi.real, bounds_error=False, fill_value=0.0)(x_int)
        psi_im = interp1d(x_grid_psi, psi.imag, bounds_error=False, fill_value=0.0)(x_int)
        psi_int = psi_re + 1j * psi_im
    else:
        psi_int = interp1d(x_grid_psi, psi, bounds_error=False, fill_value=0.0)(x_int)

    W = wigner_from_psi(psi_int, x_int, p_int, hbar=hbar)
    _check_wigner_norm(name, W, x_int, p_int)
    return _finalize_state(
        name=name, rs=rs, W=W, x_int=x_int, p_int=p_int,
        window=window, hbar=hbar,
        window_margin=window_margin, cell_center_x=cell_center_x,
    )


# Legacy alias preserved for any external callers.
build_state = build_state_from_psi


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_integration_grids(
    rs: RSGeometry,
    n_x: int,
    n_p: int,
    x_pad_factor: float,
    p_pad: float,
    window_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integration grids in (x, p), wide enough that ψ has decayed at the
    boundary and kernel tails aren't clipped during convolution.
    Centered on ⟨x⟩ so asymmetric states aren't crammed against an edge.
    """
    auto_half_width = (1.0 + window_margin) * max(rs.Delta_x, rs.Delta_p)
    x_half = max(auto_half_width * x_pad_factor, 3.0 * rs.Delta_x)
    p_half = max(auto_half_width + p_pad, 3.0 * rs.Delta_p)
    x_int = np.linspace(rs.x_mean - x_half, rs.x_mean + x_half, n_x)
    p_int = np.linspace(-p_half, p_half, n_p)
    return x_int, p_int


def _check_wigner_norm(
    name: str, W: np.ndarray, x_int: np.ndarray, p_int: np.ndarray,
) -> None:
    norm = wigner_norm(W, x_int, p_int)
    if abs(norm - 1.0) > 5e-3:
        import warnings
        warnings.warn(
            f"State {name!r}: Wigner norm = {norm:.4f} (expected 1). "
            "Consider widening the integration grid."
        )


def _finalize_state(
    *,
    name: str,
    rs: RSGeometry,
    W: np.ndarray,
    x_int: np.ndarray,
    p_int: np.ndarray,
    window: DisplayWindow,
    hbar: float,
    window_margin: float,
    cell_center_x: float | None,
) -> State:
    """Common downstream pipeline: cell anchor + window + ticks + State."""
    from .ticks import nice_ticks_around

    # Cell-overlay anchor at the location of max |W(x, 0)|, where the
    # cell's resolution argument is visually most legible. For symmetric
    # states multiple locations can share the same |W| max (e.g. 3-cat,
    # double-well): we anchor at the midpoint of all such candidates,
    # weighted by distance from ⟨x⟩, so the cell sits at the natural
    # symmetry point of the state.
    ip0 = int(np.argmin(np.abs(p_int)))
    W_at_p0 = W[:, ip0]
    if cell_center_x is None:
        abs_W = np.abs(W_at_p0)
        tol = 1e-6 * abs_W.max()
        candidates = np.where(abs_W >= abs_W.max() - tol)[0]
        cell_center_x_resolved = float(np.mean(x_int[candidates]))
    else:
        cell_center_x_resolved = cell_center_x

    # --- Display window: three constraints ---
    #   1. Cell A must not be clipped.
    #   2. Panel is square in data units (x_lim = p_lim).
    #   3. Window centers on the state's "interesting extent": midpoint of
    #      |W(x, 0)| support above 5% of peak. For symmetric states this
    #      is x=0; for asymmetric ones (Morse) it lands at the orbit
    #      center, filling the panel evenly.
    # Constraints 1 and 3 can conflict; clamp center so cell stays inside.
    half_width = (1.0 + window_margin) * max(rs.Delta_x, rs.Delta_p)

    abs_W = np.abs(W_at_p0)
    threshold = 0.05 * abs_W.max()
    significant = abs_W > threshold
    if significant.any():
        ix_lo = int(np.argmax(significant))
        ix_hi = int(len(significant) - 1 - np.argmax(significant[::-1]))
        state_extent_center = 0.5 * (float(x_int[ix_lo]) + float(x_int[ix_hi]))
    else:
        state_extent_center = cell_center_x_resolved

    if window.x_lim > 0:
        x_lim_resolved = window.x_lim
        x_center_resolved = window.x_center
    else:
        x_lim_resolved = half_width
        cell_left  = cell_center_x_resolved - rs.Delta_x
        cell_right = cell_center_x_resolved + rs.Delta_x
        min_center = cell_right - half_width
        max_center = cell_left  + half_width
        x_center_resolved = max(min_center, min(state_extent_center, max_center))

    if window.p_lim > 0:
        p_lim_resolved = window.p_lim
        p_center_resolved = window.p_center
    else:
        p_lim_resolved = half_width
        p_center_resolved = 0.0

    x_ticks = window.x_ticks or nice_ticks_around(
        x_center_resolved, x_lim_resolved, target_count=4,
    )
    p_ticks = window.p_ticks or nice_ticks_around(
        p_center_resolved, p_lim_resolved, target_count=4,
    )

    final_window = DisplayWindow(
        x_lim=x_lim_resolved, p_lim=p_lim_resolved,
        x_center=x_center_resolved, p_center=p_center_resolved,
        x_ticks=x_ticks, p_ticks=p_ticks,
    )

    return State(
        name=name,
        rs=rs,
        hbar=hbar,
        window=final_window,
        cell_center_x=cell_center_x_resolved,
        W=W,
        x_int=x_int,
        p_int=p_int,
    )
