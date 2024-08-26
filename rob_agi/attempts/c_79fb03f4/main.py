from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79fb03f4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Creating a "blue aura" around initial blue (1) squares and barriers (5 or 8)
    2. Extending blue lines horizontally and vertically up to 2 cells from barriers or edges
    3. Filling the area around barriers within a 2-cell radius
    4. Converting all cells in the "blue aura" to blue (1)
    5. Respecting barriers and grid edges

    The function creates an aura effect around blue squares and barriers,
    maintaining symmetry and following the "2 cells away" rule.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    aura = [[False for _ in range(cols)] for _ in range(rows)]

    def is_barrier(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) in [5, 8]

    def mark_aura(r: int, c: int):
        if 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) == 0:
            aura[r][c] = True

    def horizontal_expansion():
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 1:
                    for dc in [-1, 1]:
                        for i in range(1, 3):
                            nc = c + i * dc
                            if 0 <= nc < cols and not is_barrier(r, nc):
                                mark_aura(r, nc)
                            else:
                                break

    def vertical_expansion():
        for r in range(rows):
            for c in range(cols):
                if aura[r][c] or grid.get_cell(r, c) == 1:
                    for dr in [-1, 1]:
                        for i in range(1, 3):
                            nr = r + i * dr
                            if 0 <= nr < rows and not is_barrier(nr, c):
                                mark_aura(nr, c)
                            else:
                                break

    def barrier_aura():
        for r in range(rows):
            for c in range(cols):
                if is_barrier(r, c):
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            mark_aura(r + dr, c + dc)

    def fill_aura():
        for r in range(rows):
            for c in range(cols):
                if aura[r][c] and grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 1)

    horizontal_expansion()
    vertical_expansion()
    barrier_aura()
    fill_aura()

    return grid
