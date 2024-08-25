from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_6f473927(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it and adding a complementary sky blue pattern.
    
    The function analyzes the red pattern in the input grid, determines the expansion
    direction, calculates the new dimensions, and creates a new grid with the original
    pattern preserved and a complementary sky blue pattern added.
    
    Steps:
    1. Analyze the input grid to find the red pattern boundaries.
    2. Determine the expansion direction based on the red pattern's position.
    3. Calculate the new dimensions for the expanded grid.
    4. Create the expanded grid and copy the original pattern.
    5. Add the complementary sky blue pattern.
    6. Ensure proper separation between red and sky blue areas.
    
    Returns:
    ColoredGrid: The transformed grid with the original red pattern and new sky blue pattern.
    """
    rows, cols = input_grid.get_dimensions()
    red_bounds = find_red_bounds(input_grid)
    expansion_direction = determine_expansion_direction(red_bounds, rows, cols)
    new_dimensions = calculate_new_dimensions(red_bounds, rows, cols, expansion_direction)
    
    new_grid = create_expanded_grid(input_grid, new_dimensions, expansion_direction)
    add_sky_blue_pattern(new_grid, red_bounds, expansion_direction)
    
    return new_grid

def find_red_bounds(grid: ColoredGrid) -> Tuple[int, int, int, int]:
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

def determine_expansion_direction(bounds: Tuple[int, int, int, int], rows: int, cols: int) -> str:
    """Determine the direction of expansion based on the red pattern's position."""
    left, right, top, bottom = bounds
    center_col = (left + right) / 2
    center_row = (top + bottom) / 2
    
    if center_col < cols / 2:
        return "right"
    else:
        return "left"

def calculate_new_dimensions(bounds: Tuple[int, int, int, int], rows: int, cols: int, direction: str) -> Tuple[int, int]:
    """Calculate the new dimensions for the expanded grid."""
    left, right, top, bottom = bounds
    
    if direction == "right":
        new_cols = cols + (cols - right - 1) * 2 + 1
    else:  # left
        new_cols = cols + left * 2 + 1
    
    return rows, new_cols

def create_expanded_grid(input_grid: ColoredGrid, new_dimensions: Tuple[int, int], direction: str) -> ColoredGrid:
    """Create the expanded grid and copy the original pattern."""
    rows, new_cols = new_dimensions
    new_values = [[0 for _ in range(new_cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(len(input_grid.values[0])):
            if direction == "right":
                new_values[r][c] = input_grid.get_cell(r, c)
            else:  # left
                new_values[r][new_cols - len(input_grid.values[0]) + c] = input_grid.get_cell(r, c)
    
    return ColoredGrid(values=new_values)

def add_sky_blue_pattern(grid: ColoredGrid, red_bounds: Tuple[int, int, int, int], direction: str):
    """Add the complementary sky blue pattern to the expanded grid."""
    rows, cols = grid.get_dimensions()
    left, right, top, bottom = red_bounds
    
    if direction == "right":
        for r in range(rows):
            for c in range(right + 2, cols):
                if all(grid.get_cell(r, i) == 0 for i in range(right + 1, c)):
                    grid.set_cell(r, c, 8)  # Sky blue
    else:  # left
        for r in range(rows):
            for c in range(left - 1, -1, -1):
                if all(grid.get_cell(r, i) == 0 for i in range(c + 1, left)):
                    grid.set_cell(r, c, 8)  # Sky blue
