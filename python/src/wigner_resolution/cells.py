"""Phase-space cells: the Heisenberg cell A, the bitangent blobs a_θ,
and the quorum cell ã.

A state's covariance defines its Heisenberg cell

    A = π Δx Δp  ≥  h/2,

the outer ellipse with capacity semi-axes (Δx, Δp), floored at the
quantum of action by the uncertainty principle.

The symplectic polar dual of the Heisenberg cell is the quorum cell

    ã = π δx δp  ≤  h/2,

an axis-aligned ellipse with semi-axes

    (δx, δp) = (ℏ/Δp, ℏ/Δx),

reciprocal to the Heisenberg cell at action h/2: A · ã = (h/2)².
Reference: M. de Gosson and C. de Gosson, Symmetry 14, 1890 (2022).

Bitangent to both A (from within) and ã (from without) is a
one-parameter family of de Gosson quantum blobs a_θ of fixed action
h/2, parameterized by the family-orientation angle θ. The semi-axes
r_∥(θ), r_⊥(θ) of a_θ are reciprocal in ℏ,

    r_∥(θ) r_⊥(θ) = ℏ.

The family is constructed in the affine-normalized frame where A
becomes the unit disk and ã becomes the inscribed concentric circle
of radius

    r̃ = ℏ/(Δx Δp).

In that frame, a_θ is the rigid rotation by θ of a single ellipse
with semi-axes (1, r̃), bitangent to the unit disk and to the
inscribed circle. Pulled back to the original (x, p) frame via the
inverse of T(x, p) = (x/Δx, p/Δp), the family deforms across θ at
constant action h/2 while maintaining double bitangency to A and ã.

At the principal angles θ = 0 and θ = π/2, the blob's principal axes
align with the (x, p) coordinate axes, and the semi-axes recover the
Zurek scales δx = ℏ/Δp and δp = ℏ/Δx. At intermediate angles, the
blob's principal axes are tilted off Euclidean θ; the orientation
parameter θ continuously interpolates between the two principal
endpoints. Reference for Zurek's scales: W. H. Zurek, Nature 412,
712 (2001).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeisenbergCell:
    """The state's outer extent A in phase space.

    Capacity semi-axes Δx, Δp along the principal axes
    (Robertson-Schrödinger cross-term assumed zero, which holds for
    the symmetric states in the manuscript).
    """

    Delta_x: float
    Delta_p: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def area(self) -> float:
        """A = π Δx Δp."""
        return np.pi * self.Delta_x * self.Delta_p

    def __post_init__(self) -> None:
        if self.Delta_x <= 0 or self.Delta_p <= 0:
            raise ValueError("Cell widths must be positive.")


@dataclass(frozen=True)
class BitangentBlob:
    """One member a_θ of the bitangent family.

    A de Gosson quantum blob of action h/2, bitangent to the Heisenberg
    cell A from within and to the quorum cell ã from without. Labeled
    by the family-orientation angle θ.

    Geometric attributes in the original (x, p) frame:
      r_parallel: longer principal semi-axis (the "parallel" axis).
      r_perp:     shorter principal semi-axis (the "perpendicular" axis).
      principal_angle: Euclidean angle (radians) of the r_parallel axis
                       from the +x-axis, in [0, π).

    The semi-axes are reciprocal in ℏ: r_∥ · r_⊥ = ℏ. At θ ∈ {0, π/2},
    principal_angle equals θ exactly. At intermediate angles, the affine
    pullback from the disk frame tilts the principal axes off θ.
    """

    theta: float
    r_parallel: float
    r_perp: float
    principal_angle: float
    center: tuple[float, float] = (0.0, 0.0)
    hbar: float = 1.0

    @property
    def area(self) -> float:
        """a_θ = π r_∥ r_⊥ = h/2 (always)."""
        return np.pi * self.r_parallel * self.r_perp

    def __post_init__(self) -> None:
        product = self.r_parallel * self.r_perp
        if not np.isclose(product, self.hbar, rtol=1e-10):
            raise ValueError(
                f"Reciprocal-axes identity violated: "
                f"r_∥·r_⊥ = {product:.6g} ≠ ℏ = {self.hbar}"
            )


@dataclass(frozen=True)
class QuorumCell:
    """The quorum cell ã with semi-axes (δx, δp) = (ℏ/Δp, ℏ/Δx).

    Symplectic polar dual of the Heisenberg cell, bitangent to every
    blob of the family from within. An axis-aligned ellipse with no
    rotation parameter, not itself a member of the bitangent family.
    Its semi-axes are Zurek's interference scales.

    Reference for polar duality: M. de Gosson and C. de Gosson,
    Symmetry 14, 1890 (2022). Reference for Zurek's scales:
    W. H. Zurek, Nature 412, 712 (2001).
    """

    delta_x: float
    delta_p: float
    center: tuple[float, float] = (0.0, 0.0)
    hbar: float = 1.0

    @property
    def area(self) -> float:
        """ã = π δx δp = π ℏ² / (Δx Δp)."""
        return np.pi * self.delta_x * self.delta_p

    def __post_init__(self) -> None:
        if self.delta_x <= 0 or self.delta_p <= 0:
            raise ValueError("Quorum-cell semi-axes must be positive.")


def bitangent_blob_at(
    theta: float,
    heisenberg: HeisenbergCell,
    hbar: float = 1.0,
) -> BitangentBlob:
    """The bitangent blob a_θ at family-orientation angle θ.

    Construction (affine pullback):
      1. In the disk frame where A is the unit disk and ã is the
         inscribed circle of radius r̃ = ℏ/(Δx Δp), the blob a_θ is
         the rigid rotation by θ of an ellipse with semi-axes (1, r̃).
      2. Pulled back to the original (x, p) frame, the blob has
         inverse-covariance matrix M_θ = D⁻¹ R_θ diag(1, 1/r̃²) R_θᵀ D⁻¹
         with D = diag(Δx, Δp).
      3. Principal-axis lengths and orientation are recovered from the
         eigendecomposition of M_θ.

    At principal angles θ ∈ {0, π/2}, the principal axes coincide with
    the coordinate axes and r_parallel reduces to Δx or Δp respectively.
    """
    Dx, Dp = heisenberg.Delta_x, heisenberg.Delta_p
    r_tilde = hbar / (Dx * Dp)

    # Disk-frame inverse-covariance matrix, rotated by theta
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    M_disk = R @ np.diag([1.0, 1.0 / r_tilde**2]) @ R.T

    # Pull back to original (x, p) frame
    D_inv = np.diag([1.0 / Dx, 1.0 / Dp])
    M_orig = D_inv @ M_disk @ D_inv

    # Eigendecomposition: smaller eigenvalue ↔ larger semi-axis
    eigs, vecs = np.linalg.eigh(M_orig)
    r_par = 1.0 / np.sqrt(eigs[0])
    r_perp = 1.0 / np.sqrt(eigs[1])

    # Principal angle: direction of the major-axis eigenvector, in [0, π)
    major_dir = vecs[:, 0]
    alpha = np.arctan2(major_dir[1], major_dir[0])
    if alpha < 0:
        alpha += np.pi
    if alpha >= np.pi:
        alpha -= np.pi

    return BitangentBlob(
        theta=theta,
        r_parallel=r_par,
        r_perp=r_perp,
        principal_angle=alpha,
        center=heisenberg.center,
        hbar=hbar,
    )


def blob_a_pi_half(heisenberg: HeisenbergCell, hbar: float = 1.0) -> BitangentBlob:
    """The bitangent blob a_{π/2}: the family member at θ = π/2.

    Semi-axes (r_∥, r_⊥) = (Δp, δx) = (Δp, ℏ/Δp), principal angle π/2.
    """
    return bitangent_blob_at(np.pi / 2, heisenberg, hbar)


def blob_a_zero(heisenberg: HeisenbergCell, hbar: float = 1.0) -> BitangentBlob:
    """The bitangent blob a_{θ=0}: the family member at θ = 0.

    Semi-axes (r_∥, r_⊥) = (Δx, δp) = (Δx, ℏ/Δx), principal angle 0.
    """
    return bitangent_blob_at(0.0, heisenberg, hbar)


def quorum_cell(heisenberg: HeisenbergCell, hbar: float = 1.0) -> QuorumCell:
    """The quorum cell ã with semi-axes (ℏ/Δp, ℏ/Δx).

    The symplectic polar dual of `heisenberg`, bitangent to every blob
    in the bitangent family. Area ã = π ℏ²/(Δx Δp), the resolution of
    the convolved portrait W̃.
    """
    return QuorumCell(
        delta_x=hbar / heisenberg.Delta_p,
        delta_p=hbar / heisenberg.Delta_x,
        center=heisenberg.center,
        hbar=hbar,
    )