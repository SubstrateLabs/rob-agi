from rob_agi.colored_grid import ColoredGrid
from typing import Tuple

def solve_6f473927(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it and adding a complementary sky blue pattern.
    
    Steps:
    1. Analyze the input grid to find the boundaries of the red pattern.
    2. Create an expanded grid with double the width minus 1.
    3. Copy the red pattern to the right side of the new grid.
    4. Add a complementary sky blue pattern in a zigzag manner on the left side.
    
    Returns:
    ColoredGrid: The transformed grid with the original red pattern on the right and new sky blue pattern on the left.
    """
    red_bounds = find_red_boundaries(input_grid)
    new_grid = create_expanded_grid(input_grid)
    add_sky_blue_pattern(new_grid, red_bounds)
    return new_grid

def find_red_boundaries(grid: ColoredGrid) -> Tuple[int, int, int, int]:
    """Find the boundaries of the red pattern."""
    rows, cols = grid.get_dimensions()
    left, right, top, bottom = cols, -1, rows, -1
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:  # Red
                left = min(left, c)
                right = max(right, c)
                top = min(top, r)
                bottom = max(bottom, r)
    
    return left, right, top, bottom

def create_expanded_grid(input_grid: ColoredGrid) -> ColoredGrid:
    """Create the expanded grid and copy the original pattern to the right side."""
    rows, cols = input_grid.get_dimensions()
    new_cols = (cols * 2) - 1
    new_values = [[0 for _ in range(new_cols)] for _ in range(rows)]
    
    offset = new_cols - cols
    for r in range(rows):
        for c in range(cols):
            new_values[r][c + offset] = input_grid.get_cell(r, c)
    
    return ColoredGrid(values=new_values)

def add_sky_blue_pattern(grid: ColoredGrid, red_bounds: Tuple[int, int, int, int]):
    """Add the complementary sky blue pattern to the left side of the expanded grid in a zigzag manner."""
    rows, cols = grid.get_dimensions()
    left, _, _, _ = red_bounds
    
    start_col = 0
    for row in range(rows):
        if row % 2 == 1:
            start_col = max(start_col - 1, 0)
        
        col = start_col
        while col < left + (cols - 1) // 2:
            if grid.get_cell(row, col) == 0:
                grid.set_cell(row, col, 8)  # Sky blue
            col += 1
        
        start_col = min(start_col + 1, left + (cols - 1) // 2 - 1)
