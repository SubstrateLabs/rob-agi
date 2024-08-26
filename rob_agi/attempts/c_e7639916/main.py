from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e7639916(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e7639916 challenge by drawing a blue (1) rectangle that encompasses all sky-colored (8) cells.
    
    The solution follows these steps:
    1. Identify all sky (8) cells in the input grid.
    2. If fewer than 2 sky cells, return the input grid unchanged.
    3. Find the bounding rectangle that encompasses all sky cells.
    4. Draw blue (1) lines to form the rectangle, without overwriting sky cells.
    
    Returns a new ColoredGrid with the solution.
    """
    # Step 1: Identify sky cells
    sky_cells = [(r, c) for r in range(input_grid.num_rows) for c in range(input_grid.num_cols) if input_grid[r][c] == 8]
    
    # Step 2: Check if there are enough sky cells
    if len(sky_cells) < 2:
        return input_grid
    
    # Step 3: Find the bounding rectangle
    min_row = min(r for r, _ in sky_cells)
    max_row = max(r for r, _ in sky_cells)
    min_col = min(c for _, c in sky_cells)
    max_col = max(c for _, c in sky_cells)
    
    # Step 4: Draw blue lines
    output_grid = input_grid.deep_copy()
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            if row in (min_row, max_row) or col in (min_col, max_col):
                if output_grid[row][col] != 8:
                    output_grid.values[row][col] = 1  # Set to blue
    
    return output_grid
