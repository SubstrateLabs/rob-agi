from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_292dd178(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the space around blue (1) shapes with red (2).
    
    The function performs the following steps:
    1. Identifies connected blue regions in the grid.
    2. For each blue region:
       a. Determines its bounding box.
       b. Fills the bounding box with red, except for blue cells.
       c. Extends red lines from sides of the bounding box that don't contain blue cells.
    3. Ensures that originally blue cells remain blue.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with red fill around blue shapes.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def extend_red(x: int, y: int, dx: int, dy: int):
        while 0 <= x < rows and 0 <= y < cols and grid.values[x][y] != 1:
            grid.values[x][y] = 2
            x += dx
            y += dy
    
    blue_regions = grid.find_connected_regions(1)
    
    for region in blue_regions:
        min_x = min(cell[0] for cell in region)
        max_x = max(cell[0] for cell in region)
        min_y = min(cell[1] for cell in region)
        max_y = max(cell[1] for cell in region)
        
        # Fill bounding box
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if grid.values[x][y] != 1:
                    grid.values[x][y] = 2
        
        # Extend red lines
        if all(cell[0] != min_x for cell in region):
            for y in range(min_y, max_y + 1):
                extend_red(min_x - 1, y, -1, 0)
        if all(cell[0] != max_x for cell in region):
            for y in range(min_y, max_y + 1):
                extend_red(max_x + 1, y, 1, 0)
        if all(cell[1] != min_y for cell in region):
            for x in range(min_x, max_x + 1):
                extend_red(x, min_y - 1, 0, -1)
        if all(cell[1] != max_y for cell in region):
            for x in range(min_x, max_x + 1):
                extend_red(x, max_y + 1, 0, 1)
    
    # Ensure originally blue cells remain blue
    for x in range(rows):
        for y in range(cols):
            if input_grid.values[x][y] == 1:
                grid.values[x][y] = 1
    
    return grid
