from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e2092e0c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending an existing gray 'L' shape.
    
    The solution follows these steps:
    1. Identify the existing gray 'L' shape and its corner point.
    2. Calculate the extension dimensions based on 2/3 of the grid size.
    3. Extend the 'L' shape into a larger rectangle starting from the corner point.
    4. Fill the entire extension area with gray (5), including any non-gray cells within the original 'L'.
    5. Preserve the rest of the grid outside the extension area.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with an extended gray shape.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def find_original_L(grid):
        width = next(i for i, val in enumerate(grid.values[0]) if val != 5)
        height = next(i for i, row in enumerate(grid.values) if row[0] != 5)
        return width, height
    
    def find_corner_point(grid, l_width, l_height):
        for row in range(l_height):
            if grid.values[row][l_width] != 5:
                return row, l_width
        return l_height - 1, l_width - 1
    
    def calculate_extension(grid_dim, l_dim, corner_pos):
        target = round(grid_dim * 2/3)
        return max(target - corner_pos, l_dim)
    
    # Find original L and corner point
    orig_width, orig_height = find_original_L(output_grid)
    corner_row, corner_col = find_corner_point(output_grid, orig_width, orig_height)
    
    # Calculate extension dimensions
    ext_width = calculate_extension(cols, orig_width, corner_col)
    ext_height = calculate_extension(rows, orig_height, corner_row)
    
    # Extend and fill
    for row in range(corner_row, min(corner_row + ext_height, rows)):
        for col in range(corner_col, min(corner_col + ext_width, cols)):
            output_grid.set_cell(row, col, 5)
    
    return output_grid
