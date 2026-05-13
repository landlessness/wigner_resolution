"""Cat states: coherent superpositions of n coherent states in phase space.

Four configurations, all sized so the most distant lobe sits at radius
≈ p_max·√2 from the origin in the worst case:

  * n=2:        lobes at (0, ±p_max). North–south, separation 2·p_max.
  * n=3:        equilateral triangle: apex (0, +p_max), base
                (±2·p_max/√3, −p_max). Adjacent-lobe distance is
                2·p_max·√(4/3) ≈ 2.309·p_max — equilateral, but the
                side is not 2·p_max.
  * n=4 (diag): square at corners (±p_max, ±p_max). Side 2·p_max.
  * n=4 (axis): 45° rotation of diag — lobes at (±p_max√2, 0) and
                (0, ±p_max√2), forming a square of side 2·p_max
                rotated by π/4. This is Zurek's compass state.

The wavefunction is a sum of coherent states ``qt.coherent(N, αₖ)`` with
``αₖ = (qₖ + i pₖ)/√2`` placed at each lobe.

Default ``p_max = 5`` gives lobes well-separated for the Fock truncation
``N = 80`` we use here; the most extreme lobe (axis variant) has
|α|² = 50, comfortably inside the basis.

Reference: Schleich, *Quantum Optics in Phase Space* (Wiley 2001) Ch. 7;
Zurek, Nature 412, 712 (2001) for the original compass state.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from ..state import DisplayWindow, State, build_state_from_qobj

CAT_P_MAX_DEFAULT = 5.0
CAT_N_DEFAULT = 80


def cat_lobe_positions(
    n_cats: int,
    variant: str = "diag",
    p_max: float = CAT_P_MAX_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Lobe positions (q, p) for an n-cat configuration.

    See module docstring for the exact geometry of each configuration
    and the resulting adjacent-lobe distances.
    """
    if n_cats == 2:
        return np.array([0.0, 0.0]), np.array([p_max, -p_max])
    if n_cats == 3:
        q_b = 2 * p_max / np.sqrt(3)
        return (
            np.array([0.0, q_b, -q_b]),
            np.array([p_max, -p_max, -p_max]),
        )
    if n_cats == 4 and variant == "diag":
        return (
            np.array([p_max, -p_max, -p_max, p_max]),
            np.array([p_max, p_max, -p_max, -p_max]),
        )
    if n_cats == 4 and variant == "axis":
        r = p_max * np.sqrt(2)
        return (
            np.array([0.0, r, 0.0, -r]),
            np.array([r, 0.0, -r, 0.0]),
        )
    raise ValueError(
        f"Unsupported configuration: n_cats={n_cats}, variant={variant!r}"
    )


def cat_state(
    n_cats: int,
    *,
    variant: str = "diag",
    p_max: float = CAT_P_MAX_DEFAULT,
    name: str | None = None,
    hbar: float = 1.0,
    N: int = CAT_N_DEFAULT,
) -> State:
    """Build an n-cat state as a coherent superposition of QuTiP coherent states.

    ``variant`` matters only for n_cats=4; choose ``"diag"`` (compass on
    the diagonals) or ``"axis"`` (compass on the axes — Zurek 2001).
    """
    qs, ps = cat_lobe_positions(n_cats, variant=variant, p_max=p_max)
    alphas = (qs + 1j * ps) / np.sqrt(2)

    psi = sum(qt.coherent(N, a) for a in alphas)
    psi = psi.unit()

    label = f"cat_n{n_cats}"
    if n_cats == 4:
        label = f"cat_n4_{variant}"

    return build_state_from_qobj(
        name=name or label,
        qobj=psi,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )
