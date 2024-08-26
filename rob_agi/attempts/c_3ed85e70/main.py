from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def expand_square(grid: ColoredGrid, row: int, col: int, size: int) -> None:
    color = grid.values[row][col]
    for r in range(row-1, row+size+1):
        for c in range(col-1, col+size+1):
            if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                if r == row-1 or r == row+size or c == col-1 or c == col+size:
                    grid.values[r][c] = 1  # Blue border
                else:
                    grid.values[r][c] = color

def solve_3ed85e70(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Expand 2x2 colored squares to 4x4 with blue (1) borders.
    2. Expand 3x3 colored squares to 5x5 without changing their color.
    3. Preserve green (3) areas without changes.
    4. Repeat until no more changes can be made.
    """
    grid = input_grid.deep_copy()
    
    while True:
        original = grid.deep_copy()
        
        for r in range(grid.num_rows - 1):
            for c in range(grid.num_cols - 1):
                if grid.values[r][c] != 3:  # Not green
                    # Check for 2x2 square
                    if all(grid.values[r+dr][c+dc] == grid.values[r][c] for dr in range(2) for dc in range(2)):
                        expand_square(grid, r, c, 2)
                    # Check for 3x3 square
                    elif r < grid.num_rows - 2 and c < grid.num_cols - 2 and \
                         all(grid.values[r+dr][c+dc] == grid.values[r][c] for dr in range(3) for dc in range(3)):
                        expand_square(grid, r, c, 3)
        
        if grid.values == original.values:
            break
    
    return grid
