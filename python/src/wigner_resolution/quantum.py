"""Quantum state geometry: Robertson-Schrödinger covariance from a sampled ψ(x).

The currency of the paper is *action*. We work in natural units where ℏ = 1,
m = 1, ω = 1 throughout. In these units:

  * the quantum of action h/2 reduces to π
  * the action capacity of a state is A = π Δx Δp, expressed as a multiple
    of h/2 by the ratio A/(h/2) = Δx Δp / ℏ
  * a Heisenberg-saturated state has A/(h/2) = 1
  * the harmonic oscillator state |n⟩ has Δx = Δp = √(2n+1) and
    A/(h/2) = 2n+1

For all states in the manuscript the off-diagonal moment ⟨xp + px⟩/2 vanishes
by parity or by the symmetric phase-space placement of cat lobes, so this
module returns only the diagonal entries (Δx, Δp).

Reference: Robertson Phys. Rev. 34, 163 (1929); Schrödinger Sitzungsber.
Preuss. Akad. Wiss. Berlin 24, 296 (1930).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# Tolerance for the Robertson-Schrödinger inequality check. Analytic ground
# states saturate the bound exactly; the finite-difference ⟨p²⟩ underestimates
# the continuous value by O(dx²), so we need a relative tolerance that absorbs
# the discretization error. 1e-3 is comfortable for dx ≤ 0.01.
_RS_TOLERANCE = 1e-3


@dataclass(frozen=True)
class RSGeometry:
    """Robertson-Schrödinger geometry of a quantum state."""

    Delta_x: float
    Delta_p: float
    x_mean: float
    hbar: float

    @property
    def A_over_h_half(self) -> float:
        """Extended-cell action in units of h/2. Equals Δx Δp / ℏ."""
        return (self.Delta_x * self.Delta_p) / self.hbar


def numerical_covariance(
    psi: np.ndarray,
    x_grid: np.ndarray,
    hbar: float = 1.0,
) -> RSGeometry:
    """RS geometry of a wavefunction sampled on a uniform x grid.

    ⟨x⟩, ⟨x²⟩ from |ψ|²; ⟨p²⟩ from ℏ² ∫|dψ/dx|² dx (valid for normalizable ψ
    where boundary terms vanish). σ_xp is taken to be zero — exact for all
    states in the manuscript.
    """
    dx = float(x_grid[1] - x_grid[0])
    # Re-normalize so the integral works in absolute terms.
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm <= 0:
        raise ValueError("ψ has zero norm.")
    psi_n = psi / norm
    prob = np.abs(psi_n) ** 2

    x_mean = float(np.sum(x_grid * prob) * dx)
    sigma_xx = float(np.sum((x_grid - x_mean) ** 2 * prob) * dx)

    # ⟨p²⟩ = ℏ² ∫ |dψ/dx|² dx, evaluated by finite differences.
    dpsi = np.diff(psi_n) / dx
    sigma_pp = float(hbar ** 2 * np.sum(np.abs(dpsi) ** 2) * dx)

    # RS inequality (with σ_xp = 0): σ_xx σ_pp ≥ (ℏ/2)².
    if sigma_xx * sigma_pp < (hbar / 2) ** 2 * (1 - _RS_TOLERANCE):
        raise ValueError(
            f"Robertson-Schrödinger inequality violated: "
            f"σ_xx σ_pp = {sigma_xx * sigma_pp:.6e} < (ℏ/2)² = {(hbar / 2) ** 2:.6e}"
        )

    return RSGeometry(
        Delta_x=float(np.sqrt(2 * sigma_xx)),
        Delta_p=float(np.sqrt(2 * sigma_pp)),
        x_mean=x_mean,
        hbar=hbar,
    )


@dataclass(frozen=True)
class SchrodingerSolution:
    """Output of solve_schrodinger: eigenvalues plus L²-normalized ψ_n on x_grid."""

    energies: np.ndarray         # shape (n_states,), ascending
    psi_matrix: np.ndarray       # shape (len(x_grid), n_states); column n is ψ_n
    x_grid: np.ndarray
    dx: float

    def psi(self, n: int) -> np.ndarray:
        """The n-th eigenstate (0-indexed)."""
        if n < 0 or n >= len(self.energies):
            raise IndexError(f"n={n} out of range [0, {len(self.energies)})")
        return self.psi_matrix[:, n]


def solve_schrodinger(
    V: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    *,
    dx: float = 0.01,
    n_states: int = 10,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> SchrodingerSolution:
    """Time-independent Schrödinger eigenvalue problem on a uniform grid.

    H ψ = E ψ with H = -ℏ²/(2m) d²/dx² + V(x). Finite-difference (3-point
    stencil), then `scipy.sparse.linalg.eigsh` for the lowest n_states.

    The grid boundaries must be chosen so V at the edges substantially
    exceeds the highest eigenvalue of interest, so the implicit Dirichlet
    boundary doesn't leak.
    """
    x_grid = np.arange(x_min, x_max + dx / 2, dx)
    nx = len(x_grid)
    V_vec = V(x_grid)

    # 3-point Laplacian: -ψ''(x) ≈ -(ψ_{i-1} - 2ψ_i + ψ_{i+1}) / dx²
    # Hamiltonian diagonal: ℏ²/(m dx²) + V_i
    # Hamiltonian off-diagonal: -ℏ²/(2 m dx²)
    main_diag = hbar ** 2 / (mass * dx ** 2) + V_vec
    off_diag = np.full(nx - 1, -hbar ** 2 / (2 * mass * dx ** 2))
    H = diags(
        [off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format="csr",
    )

    # sigma=v_min - 1 lets eigsh use shift-invert to grab the lowest n_states.
    energies, psi_matrix = eigsh(
        H, k=n_states, which="SM",
        v0=np.ones(nx) / np.sqrt(nx),     # deterministic starting vector
    )
    order = np.argsort(energies)
    energies = energies[order]
    psi_matrix = psi_matrix[:, order]

    # L²-normalize each eigenstate
    for j in range(n_states):
        col = psi_matrix[:, j]
        norm = np.sqrt(np.sum(np.abs(col) ** 2) * dx)
        if norm > 0:
            psi_matrix[:, j] = col / norm
        # Sign convention: ψ at the rightmost interior antinode is positive
        peak_idx = int(np.argmax(np.abs(psi_matrix[:, j])))
        if psi_matrix[peak_idx, j] < 0:
            psi_matrix[:, j] = -psi_matrix[:, j]

    return SchrodingerSolution(
        energies=energies,
        psi_matrix=psi_matrix,
        x_grid=x_grid,
        dx=dx,
    )
