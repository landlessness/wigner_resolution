"""Squeezed vacuum state.

The squeezed vacuum at parameter r is built by acting the QuTiP squeeze
operator on the vacuum:

    |ψ⟩ = S(r) |0⟩

with S(r) = exp((r/2)(a² − a†²)) in QuTiP's convention. In the package
convention Δx = √2 σ_x, this gives

    Δx = e^(−r),   Δp = e^(+r),   Δx Δp = ℏ = 1,

so the state saturates the Robertson–Schrödinger bound for every r.
"""

from __future__ import annotations

import qutip as qt

from ..state import DisplayWindow, State, build_state_from_qobj


def squeezed_vacuum_state(
    r: float = 0.5,
    *,
    name: str | None = None,
    hbar: float = 1.0,
    N: int = 80,
) -> State:
    """Squeezed vacuum at squeezing parameter ``r``.

    ``N`` is the Fock-basis truncation. The default N=80 is comfortable
    for r ≤ 1; very large r needs more.
    """
    psi = qt.squeeze(N, r) * qt.basis(N, 0)
    return build_state_from_qobj(
        name=name or f"squeezed_vacuum_r{r:g}",
        qobj=psi,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )
