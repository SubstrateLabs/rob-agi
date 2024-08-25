from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_6f473927(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it and adding a complementary sky blue pattern.
    
    The function analyzes the red pattern in the input grid, determines the expansion
    direction based on available space, calculates the new dimensions, creates a new grid
    with the original pattern preserved, and adds a complementary sky blue pattern in a
    zigzag manner.
    
    Steps:
    1. Analyze the input grid to find the red pattern boundaries and empty space.
    2. Determine the expansion direction based on available space.
    3. Calculate the new dimensions for the expanded grid.
    4. Create the expanded grid and copy the original pattern.
    5. Add the complementary sky blue pattern in a zigzag manner.
    
    Returns:
    ColoredGrid: The transformed grid with the original red pattern and new sky blue pattern.
    """
    rows, cols = input_grid.get_dimensions()
    red_bounds = find_red_bounds(input_grid)
    expansion_direction = determine_expansion_direction(red_bounds, cols)
    new_dimensions = calculate_new_dimensions(red_bounds, rows, cols)
    
    new_grid = create_expanded_grid(input_grid, new_dimensions, expansion_direction)
    add_sky_blue_pattern(new_grid, expansion_direction)
    
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

def determine_expansion_direction(bounds: Tuple[int, int, int, int], cols: int) -> str:
    """Determine the direction of expansion based on available space."""
    left, right, _, _ = bounds
    left_space = left
    right_space = cols - right - 1
    
    return "left" if left_space > right_space else "right"

def calculate_new_dimensions(bounds: Tuple[int, int, int, int], rows: int, cols: int) -> Tuple[int, int]:
    """Calculate the new dimensions for the expanded grid."""
    left, right, _, _ = bounds
    left_space = left
    right_space = cols - right - 1
    
    new_cols = cols + (2 * max(left_space, right_space)) + 1
    return rows, new_cols

def create_expanded_grid(input_grid: ColoredGrid, new_dimensions: Tuple[int, int], direction: str) -> ColoredGrid:
    """Create the expanded grid and copy the original pattern."""
    rows, new_cols = new_dimensions
    old_cols = input_grid.get_dimensions()[1]
    new_values = [[0 for _ in range(new_cols)] for _ in range(rows)]
    
    offset = 0 if direction == "right" else new_cols - old_cols
    
    for r in range(rows):
        for c in range(old_cols):
            new_values[r][c + offset] = input_grid.get_cell(r, c)
    
    return ColoredGrid(values=new_values)

def add_sky_blue_pattern(grid: ColoredGrid, direction: str):
    """Add the complementary sky blue pattern to the expanded grid in a zigzag manner."""
    rows, cols = grid.get_dimensions()
    front = 0 if direction == "left" else cols - 1
    
    for row in range(rows):
        if row % 2 == 0:
            if direction == "left":
                front = min(front + 1, cols - 1)
            else:
                front = max(front - 1, 0)
        
        fill_start = 0 if direction == "left" else front
        fill_end = front + 1 if direction == "left" else cols
        
        for col in range(fill_start, fill_end):
            if grid.get_cell(row, col) == 0:
                grid.set_cell(row, col, 8)  # Sky blue
            else:
                if direction == "left":
                    front = col - 1
                else:
                    front = col
                break
