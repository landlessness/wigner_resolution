"""Double-well oscillator eigenstates.

V(x) = -μ²/2 · x² + λ/4 · x⁴ with μ² = 4, λ = 1. Two symmetric wells at
x = ±√(μ²/λ) = ±2. Barrier at the origin with height V_0 = μ⁴/(4λ) = 4.
Harmonic frequency at each well minimum: ω_well = √2 · μ = 2√2.

For n = 5 the eigenstate sits well above the barrier — a delocalized
state straddling both wells, with rich nodal structure and strong
Wigner negativity from the inter-well interference.

The potential is symmetric under parity, so ⟨x⟩ = 0.
"""

from __future__ import annotations

import numpy as np

from ..quantum import solve_schrodinger
from ..state import DisplayWindow, State, build_state_from_psi

mu2 = 4.0
lam = 1.0

X_MIN = -6.0
X_MAX = 6.0
DX = 0.01
N_STATES = 8


def double_well_V(x: np.ndarray) -> np.ndarray:
    """V(x) = -μ²/2 · x² + λ/4 · x⁴."""
    return -mu2 / 2 * x ** 2 + lam / 4 * x ** 4


def double_well_state(
    n: int = 5,
    *,
    name: str | None = None,
    hbar: float = 1.0,
) -> State:
    """Build the n-th double-well eigenstate via finite-difference."""
    if n < 0:
        raise ValueError("n must be non-negative")

    soln = solve_schrodinger(double_well_V, X_MIN, X_MAX, dx=DX, n_states=N_STATES, hbar=hbar)
    psi = soln.psi(n)
    x_psi = soln.x_grid

    # build_state_from_psi computes the display window from W's extent.
    # The state is symmetric so it lands at x = 0.
    window = DisplayWindow(x_lim=0.0, p_lim=0.0)

    return build_state_from_psi(
        name=name or f"double_well_n{n}",
        psi=psi,
        x_grid_psi=x_psi,
        window=window,
        hbar=hbar,
    )
