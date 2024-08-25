from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_9edfc990(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rule:
    Any cell with a value of 0 that is adjacent (horizontally or vertically) 
    to a cell with a value of 1 becomes 1. This process continues until no 
    more changes can be made.

    The transformation is done in-place to improve efficiency.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    
    def get_neighbors(row: int, col: int) -> List[Tuple[int, int]]:
        return [(row+dr, col+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                if 0 <= row+dr < rows and 0 <= col+dc < cols]

    to_check: Set[Tuple[int, int]] = set((r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == 0)

    while to_check:
        next_check = set()
        for r, c in to_check:
            if input_grid.get_cell(r, c) == 0 and any(input_grid.get_cell(nr, nc) == 1 for nr, nc in get_neighbors(r, c)):
                input_grid.set_cell(r, c, 1)
                next_check.update((nr, nc) for nr, nc in get_neighbors(r, c) if input_grid.get_cell(nr, nc) == 0)
        to_check = next_check

    return input_grid
