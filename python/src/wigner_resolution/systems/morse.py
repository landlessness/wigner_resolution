"""Morse oscillator eigenstates.

V(x) = D_e (1 - exp(-α x))² with the standard dimensionless parameters
D_e = 12.5, α = 0.2. Well bottom at x = 0; infinite wall at x → -∞;
dissociation energy D_e at x → +∞. Harmonic frequency at the well bottom:
ω = α √(2 D_e) = 1.

The analytic Morse spectrum

    E_n = ω(n + 1/2) − [ω(n + 1/2)]² / (4 D_e)

terminates at n_max = floor(√(2 D_e)/α − 1/2). For D_e=12.5, α=0.2:
n_max = 24.
"""

from __future__ import annotations

import numpy as np

from ..quantum import solve_schrodinger
from ..state import DisplayWindow, State, build_state_from_psi

D_e = 12.5
alpha = 0.2

# Solver grid: wide enough that V at the boundaries dwarfs the highest
# bound-state energy.
X_MIN = -10.0
X_MAX = 20.0
DX = 0.01
N_STATES = 18  # enough headroom past n=8


def morse_V(x: np.ndarray) -> np.ndarray:
    """V(x) = D_e (1 - e^(-α x))²."""
    return D_e * (1.0 - np.exp(-alpha * x)) ** 2


def morse_energy_exact(n: int) -> float:
    """Analytic Morse spectrum."""
    omega = alpha * np.sqrt(2 * D_e)
    return omega * (n + 0.5) - (omega * (n + 0.5)) ** 2 / (4 * D_e)


def morse_state(
    n: int = 8,
    *,
    name: str | None = None,
    hbar: float = 1.0,
) -> State:
    """Build the n-th Morse eigenstate via finite-difference."""
    if n < 0:
        raise ValueError("n must be non-negative")
    n_max = int(np.floor(np.sqrt(2 * D_e) / alpha - 0.5))
    if n > n_max:
        raise ValueError(
            f"n={n} exceeds n_max={n_max} for Morse with D_e={D_e}, α={alpha}"
        )

    soln = solve_schrodinger(morse_V, X_MIN, X_MAX, dx=DX, n_states=N_STATES, hbar=hbar)

    # Sanity: numerical E_n should agree with analytic to a few × 10^-4.
    E_num = soln.energies[n]
    E_exact = morse_energy_exact(n)
    if abs(E_num - E_exact) > 5e-3:
        import warnings
        warnings.warn(
            f"Morse n={n}: numerical E={E_num:.6f}, exact={E_exact:.6f}, "
            f"diff={E_num - E_exact:+.2e}"
        )

    psi = soln.psi(n)
    x_psi = soln.x_grid

    # build_state computes the display window from W's extent. We
    # override x_ticks to match the figure's -left/0/+right tick rhythm;
    # the auto-picked ticks land at 0 and 5 with the 1-2-5 step rule for
    # this asymmetric window.
    window = DisplayWindow(
        x_lim=0.0, p_lim=0.0,
        x_ticks=(-3.0, 0.0, 6.0),
    )

    return build_state_from_psi(
        name=name or f"morse_n{n}",
        psi=psi,
        x_grid_psi=x_psi,
        window=window,
        hbar=hbar,
        # cell_center_x left as None: build_state_from_psi picks max
        # |W(x,0)|, the location of deepest negativity.
    )
