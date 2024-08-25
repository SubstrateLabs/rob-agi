from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_6f473927(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it and adding a complementary sky blue pattern.
    
    The function analyzes the red pattern in the input grid, determines the expansion
    direction based on which edge the red pattern touches, calculates the new dimensions,
    creates a new grid with the original pattern preserved, and adds a complementary
    sky blue pattern in a zigzag manner.
    
    Steps:
    1. Analyze the input grid to find which edge the red pattern touches and its boundaries.
    2. Create the expanded grid with double the width minus 1.
    3. Copy the red pattern to the appropriate side of the new grid.
    4. Add the complementary sky blue pattern in a zigzag manner from the opposite edge.
    
    Returns:
    ColoredGrid: The transformed grid with the original red pattern and new sky blue pattern.
    """
    rows, cols = input_grid.get_dimensions()
    edge, red_bounds = find_red_edge_and_bounds(input_grid)
    new_cols = (cols * 2) - 1
    
    new_grid = create_expanded_grid(input_grid, (rows, new_cols), edge)
    add_sky_blue_pattern(new_grid, edge, red_bounds)
    
    return new_grid

def find_red_edge_and_bounds(grid: ColoredGrid) -> Tuple[str, Tuple[int, int, int, int]]:
    """Find which edge the red pattern touches and its boundaries."""
    rows, cols = grid.get_dimensions()
    left, right, top, bottom = cols, -1, rows, -1
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:  # Red
                left = min(left, c)
                right = max(right, c)
                top = min(top, r)
                bottom = max(bottom, r)
    
    edge = "left" if left == 0 else "right"
    return edge, (left, right, top, bottom)

def create_expanded_grid(input_grid: ColoredGrid, new_dimensions: Tuple[int, int], edge: str) -> ColoredGrid:
    """Create the expanded grid and copy the original pattern."""
    rows, new_cols = new_dimensions
    old_cols = input_grid.get_dimensions()[1]
    new_values = [[0 for _ in range(new_cols)] for _ in range(rows)]
    
    offset = 0 if edge == "left" else new_cols - old_cols
    
    for r in range(rows):
        for c in range(old_cols):
            new_values[r][c + offset] = input_grid.get_cell(r, c)
    
    return ColoredGrid(values=new_values)

def add_sky_blue_pattern(grid: ColoredGrid, edge: str, red_bounds: Tuple[int, int, int, int]):
    """Add the complementary sky blue pattern to the expanded grid in a zigzag manner."""
    rows, cols = grid.get_dimensions()
    left, right, top, bottom = red_bounds
    
    start_col = cols - 1 if edge == "left" else 0
    step = -1 if edge == "left" else 1
    
    for row in range(rows):
        if row % 2 == 1:
            start_col += step
        
        col = start_col
        while (edge == "left" and col >= right + 1) or (edge == "right" and col <= left - 1):
            if grid.get_cell(row, col) == 0:
                grid.set_cell(row, col, 8)  # Sky blue
            col -= step
        
        if edge == "left":
            start_col = min(start_col + 1, cols - 1)
        else:
            start_col = max(start_col - 1, 0)
