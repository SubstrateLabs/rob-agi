from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e2092e0c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending an existing gray 'L' shape.
    
    The solution follows these steps:
    1. Analyze the input grid to find the dimensions and locate the existing gray 'L' shape.
    2. Identify the corner point of the 'L' shape.
    3. Calculate the target dimensions based on 2/3 of the grid size.
    4. Determine the extension dimensions, ensuring not to shrink the existing gray area.
    5. Create a new grid as a deep copy of the input grid.
    6. Extend the gray area from the corner point, filling in a rectangular shape.
    7. Ensure the extension doesn't go beyond grid boundaries.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with an extended gray shape.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def find_gray_L(grid):
        width = 0
        height = 0
        for i in range(min(rows, cols)):
            if grid.get_cell(0, i) == 5:
                width = i + 1
            if grid.get_cell(i, 0) == 5:
                height = i + 1
            if grid.get_cell(0, i) != 5 and grid.get_cell(i, 0) != 5:
                break
        return width, height
    
    def find_corner_point(grid, l_width, l_height):
        return l_height - 1, l_width - 1
    
    def calculate_extension(grid_dim, l_dim, corner_pos):
        target = round(grid_dim * 2/3)
        return max(target - corner_pos, l_dim)
    
    # Find original L and corner point
    orig_width, orig_height = find_gray_L(output_grid)
    corner_row, corner_col = find_corner_point(output_grid, orig_width, orig_height)
    
    # Calculate extension dimensions
    ext_width = calculate_extension(cols, orig_width, corner_col)
    ext_height = calculate_extension(rows, orig_height, corner_row)
    
    # Extend and fill
    for row in range(corner_row + 1 - orig_height, min(corner_row + ext_height, rows)):
        for col in range(corner_col + 1 - orig_width, min(corner_col + ext_width, cols)):
            output_grid.set_cell(row, col, 5)
    
    return output_grid
