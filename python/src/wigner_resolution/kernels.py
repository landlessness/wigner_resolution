"""The polar-dual kernel family K_θ.

    K_θ(x, p) = (1/πℏ) exp(-x_θ²/δ‖² - p_θ²/δ⊥²)

with rotated coordinates

    x_θ =  x cosθ + p sinθ
    p_θ = -x sinθ + p cosθ.

This is the Wigner function of the squeezed coherent state matched to the
extended cell's covariance and rotated to angle θ. Its action is h/2 by
construction; its non-negativity follows from being the Wigner function of
a pure state.

Implementation: a rotated 2D Gaussian with diagonal covariance
Σ₀ = diag(δ‖²/2, δ⊥²/2) in the rotated frame, then Σ_θ = R(θ) Σ₀ R(θ)ᵀ in
the unrotated phase-space frame. Evaluated via scipy.stats.multivariate_normal.

The prefactor 1/(πℏ) in the displayed kernel comes out of
scipy.stats.multivariate_normal *only* under the polar-dual condition
δ‖δ⊥ = ℏ. The assertion at the top of K_theta_mesh enforces this.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import multivariate_normal

from .cells import SqueezedCell, squeezed_cell_at, ExtendedCell

# Tolerance for the polar-duality check. The cell-building routines compute
# δ⊥ = ℏ/δ‖ from the cell's δ‖, so the product should be exact to floating-
# point — a slack tolerance still catches construction bugs without false
# positives.
_POLAR_DUAL_TOL = 1e-10


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def K_theta_mesh(
    cell: SqueezedCell,
    xx: np.ndarray,
    pp: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Evaluate K_θ on a 2D mesh (xx, pp).

    `xx` and `pp` are 2D arrays of identical shape from ``np.meshgrid``.
    Returns an array of the same shape.

    Asserts polar duality δ‖ δ⊥ = ℏ. The scipy.stats.multivariate_normal
    prefactor 1/(2π √det Σ) equals 1/(π δ‖ δ⊥), so the manuscript's
    1/(πℏ) prefactor (Eq. 9) is recovered only when this product equals ℏ.
    Violating it would silently produce a kernel that integrates to 1 but
    no longer matches the displayed equation.
    """
    product = cell.delta_parallel * cell.delta_perp
    if abs(product - hbar) > _POLAR_DUAL_TOL:
        raise ValueError(
            f"Polar duality violated: δ‖·δ⊥ = {product:.6e} ≠ ℏ = {hbar:.6e}. "
            "Build the cell via squeezed_cell_at() to ensure δ⊥ = ℏ/δ‖."
        )

    # Build Σ_θ in the unrotated phase-space frame.
    R = _rotation_matrix(cell.theta)
    Sigma_0 = np.diag([cell.delta_parallel ** 2 / 2, cell.delta_perp ** 2 / 2])
    Sigma_theta = R @ Sigma_0 @ R.T

    pos = np.dstack([xx - cell.center[0], pp - cell.center[1]])
    rv = multivariate_normal(mean=[0.0, 0.0], cov=Sigma_theta)
    return rv.pdf(pos)


def K_theta_from_extended(
    theta: float,
    extended: ExtendedCell,
    xx: np.ndarray,
    pp: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Convenience: build the squeezed cell at angle θ inscribed in `extended`,
    then evaluate K_θ on (xx, pp)."""
    cell = squeezed_cell_at(theta, extended, hbar=hbar)
    return K_theta_mesh(cell, xx, pp, hbar=hbar)
