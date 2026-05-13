"""Harmonic oscillator eigenstates.

The n-th harmonic oscillator eigenstate is ``qt.basis(N, n)`` in QuTiP's
Fock-basis representation. In natural units (ℏ = m = ω = 1) the n-th
state has Δx = Δp = √(2n+1) and action capacity A/(h/2) = 2n+1.
"""

from __future__ import annotations

import qutip as qt

from ..state import DisplayWindow, State, build_state_from_qobj


def harmonic_state(
    n: int = 1,
    *,
    name: str | None = None,
    hbar: float = 1.0,
    N: int | None = None,
) -> State:
    """Build the n-th harmonic oscillator eigenstate.

    ``N`` is the Fock-basis truncation. Defaults to ``max(40, 4*(n+1))``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if N is None:
        N = max(40, 4 * (n + 1))
    psi = qt.basis(N, n)
    return build_state_from_qobj(
        name=name or f"harmonic_n{n}",
        qobj=psi,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )
