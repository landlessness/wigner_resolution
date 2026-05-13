"""Render the four-row cats figure to tex/figures/cats.pdf."""

from __future__ import annotations

from pathlib import Path

from wigner_resolution.figures.grid import assemble_grid, save_grid
from wigner_resolution.plotstyle import use_prl_style
from wigner_resolution.systems.cats import cat_state

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "cats.pdf"

use_prl_style(use_tex=True)

states = [
    cat_state(2),
    cat_state(3),
    cat_state(4, variant="diag"),
    cat_state(4, variant="axis"),
]

for s in states:
    print(f"{s.name}: Δx={s.rs.Delta_x:.3f}, Δp={s.rs.Delta_p:.3f}, "
          f"⟨x⟩={s.rs.x_mean:.3f}, cell_x={s.cell_center_x:.3f}, "
          f"A/(h/2)={s.rs.A_over_h_half:.2f}")

fig = assemble_grid(states)
save_grid(fig, OUT)
