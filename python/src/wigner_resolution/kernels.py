"""The bitangent kernel family K_θ.

K_θ is the Wigner function of the bitangent blob a_θ: a centered
two-dimensional Gaussian of action h/2, with covariance ellipse
coincident with a_θ. Its non-negativity follows from being the Wigner
function of a pure Gaussian state.

Implementation: a rotated 2D Gaussian with diagonal covariance
Σ₀ = diag(r_∥²/2, r_⊥²/2) in the blob's principal-axis frame, then
Σ_θ = R(α) Σ₀ R(α)ᵀ in the unrotated (x, p) frame, where α is the
blob's principal-axis orientation in original coordinates. At the
principal angles θ = 0 and θ = π/2 the rotation α equals θ; at
intermediate angles α is the Euclidean angle of the blob's major
axis after the affine pullback from the disk frame.

The prefactor 1/(πℏ) of the kernel comes out of
scipy.stats.multivariate_normal *only* under the reciprocal-axes
condition r_∥ r_⊥ = ℏ. The assertion at the top of K_theta_mesh
enforces this.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import multivariate_normal

from .cells import BitangentBlob, bitangent_blob_at, HeisenbergCell

# Tolerance for the reciprocal-axes check. The blob-building routines compute
# r_⊥ = ℏ/r_∥ from the blob's r_∥, so the product should be exact to floating-
# point — a slack tolerance still catches construction bugs without false
# positives.
_RECIPROCAL_TOL = 1e-10


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def K_theta_mesh(
    blob: BitangentBlob,
    xx: np.ndarray,
    pp: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Evaluate K_θ on a 2D mesh (xx, pp).

    `xx` and `pp` are 2D arrays of identical shape from ``np.meshgrid``.
    Returns an array of the same shape.

    The covariance is built by rotating diag(r_∥²/2, r_⊥²/2) into the
    blob's principal-axis frame via R(blob.principal_angle). For the
    affine-pullback family, principal_angle equals θ at the principal
    angles {0, π/2} and is tilted off θ at intermediate angles.

    Asserts reciprocal axes r_∥ r_⊥ = ℏ. The scipy.stats.multivariate_normal
    prefactor 1/(2π √det Σ) equals 1/(π r_∥ r_⊥), so the manuscript's
    1/(πℏ) prefactor is recovered only when this product equals ℏ.
    Violating it would silently produce a kernel that integrates to 1 but
    no longer matches the displayed equation.
    """
    product = blob.r_parallel * blob.r_perp
    if abs(product - hbar) > _RECIPROCAL_TOL:
        raise ValueError(
            f"Reciprocal axes violated: r_∥·r_⊥ = {product:.6e} ≠ ℏ = {hbar:.6e}. "
            "Build the blob via bitangent_blob_at() to ensure r_⊥ = ℏ/r_∥."
        )

    # Build Σ_θ in the unrotated (x, p) frame.
    # Rotate the diagonal principal-axis covariance by the blob's principal
    # angle, NOT by the family parameter θ. Under the affine pullback these
    # differ at intermediate θ.
    R = _rotation_matrix(blob.principal_angle)
    Sigma_0 = np.diag([blob.r_parallel ** 2 / 2, blob.r_perp ** 2 / 2])
    Sigma_theta = R @ Sigma_0 @ R.T

    pos = np.dstack([xx - blob.center[0], pp - blob.center[1]])
    rv = multivariate_normal(mean=[0.0, 0.0], cov=Sigma_theta)
    return rv.pdf(pos)


def K_theta_from_heisenberg(
    theta: float,
    heisenberg: HeisenbergCell,
    xx: np.ndarray,
    pp: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Convenience: build the bitangent blob a_θ inscribed in `heisenberg`,
    then evaluate K_θ on (xx, pp)."""
    blob = bitangent_blob_at(theta, heisenberg, hbar=hbar)
    return K_theta_mesh(blob, xx, pp, hbar=hbar)