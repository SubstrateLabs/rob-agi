from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_6f473927(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it and adding a complementary sky blue pattern.
    
    Steps:
    1. Analyze the input grid to find the boundaries of the red pattern.
    2. Determine the output grid dimensions based on the red pattern position.
    3. Create an expanded grid and copy the red pattern to the appropriate side.
    4. Add a complementary sky blue pattern on the opposite side.
    5. Refine the sky blue pattern to complement the red pattern's shape.
    
    Returns:
    ColoredGrid: The transformed grid with the original red pattern and new sky blue pattern.
    """
    red_bounds = find_red_boundaries(input_grid)
    new_grid = create_expanded_grid(input_grid, red_bounds)
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

def create_expanded_grid(input_grid: ColoredGrid, red_bounds: Tuple[int, int, int, int]) -> ColoredGrid:
    """Create the expanded grid and copy the original pattern to the appropriate side."""
    rows, cols = input_grid.get_dimensions()
    left, right, _, _ = red_bounds
    red_center = (left + right) / 2
    
    new_cols = cols + max(left, cols - right - 1)
    new_values = [[0 for _ in range(new_cols)] for _ in range(rows)]
    
    if red_center < cols / 2:
        # Red pattern on the right side
        offset = new_cols - cols
    else:
        # Red pattern on the left side
        offset = 0
    
    for r in range(rows):
        for c in range(cols):
            new_values[r][c + offset] = input_grid.get_cell(r, c)
    
    return ColoredGrid(values=new_values)

def add_sky_blue_pattern(grid: ColoredGrid, red_bounds: Tuple[int, int, int, int]):
    """Add the complementary sky blue pattern to the grid."""
    rows, cols = grid.get_dimensions()
    left, right, top, bottom = red_bounds
    
    red_center = (left + right) / 2
    sky_blue_start = cols - 1 if red_center < cols / 2 else 0
    sky_blue_direction = -1 if red_center < cols / 2 else 1
    
    for row in range(rows):
        red_in_row = any(grid.get_cell(row, c) == 2 for c in range(cols))
        col = sky_blue_start
        zigzag = row % 2 == 0
        
        while 0 <= col < cols:
            if grid.get_cell(row, col) == 0:
                if red_in_row:
                    if zigzag:
                        grid.set_cell(row, col, 8)  # Sky blue
                    zigzag = not zigzag
                else:
                    if (col - sky_blue_start) % 2 == 0:
                        grid.set_cell(row, col, 8)  # Sky blue
            elif grid.get_cell(row, col) == 2:
                break
            col += sky_blue_direction
    
    refine_sky_blue_pattern(grid, red_bounds)

def refine_sky_blue_pattern(grid: ColoredGrid, red_bounds: Tuple[int, int, int, int]):
    """Refine the sky blue pattern to complement the red pattern's shape."""
    rows, cols = grid.get_dimensions()
    left, right, top, bottom = red_bounds
    
    for row in range(rows):
        red_cells = [c for c in range(cols) if grid.get_cell(row, c) == 2]
        if red_cells:
            min_red, max_red = min(red_cells), max(red_cells)
            for col in range(cols):
                if col < min_red and grid.get_cell(row, col) == 0:
                    grid.set_cell(row, col, 8)
                elif col > max_red and grid.get_cell(row, col) == 0:
                    grid.set_cell(row, col, 8)
