from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac0c5833(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding red (2) regions into 3x3 areas around yellow (4) cells,
    with one corner missing. The expansion preserves existing structures and consistently
    chooses which corner to leave empty. Yellow cells act as anchors for new red expansions,
    and existing red and yellow structures are preserved.
    """
    grid = input_grid.deep_copy()
    for row in range(grid.num_rows):
        for col in range(grid.num_cols):
            if grid.values[row][col] == 4:  # Yellow cell
                expand_around_yellow(grid, row, col)
    return grid

def expand_around_yellow(grid: ColoredGrid, row: int, col: int):
    corners = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
    empty_corner = next((c for c in corners if 0 <= c[0] < grid.num_rows and 0 <= c[1] < grid.num_cols), None)
    
    for i in range(max(0, row-1), min(grid.num_rows, row+2)):
        for j in range(max(0, col-1), min(grid.num_cols, col+2)):
            if (i, j) != (row, col) and (i, j) != empty_corner:
                if grid.values[i][j] == 0:  # Only change if it's currently black
                    grid.values[i][j] = 2  # Change to red
