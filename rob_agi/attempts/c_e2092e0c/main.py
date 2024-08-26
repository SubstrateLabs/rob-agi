from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e2092e0c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending an existing gray 'L' shape.
    
    The solution follows these steps:
    1. Identify the existing gray 'L' shape in the top-left corner.
    2. Calculate the extension width as approximately 2/3 of the grid width minus the width of the original 'L'.
    3. Extend the 'L' shape into a larger rectangle that reaches about 2/3 across the grid and down to the bottom.
    4. Fill in all vertical gaps within this rectangle with gray (5).
    5. Update the grid with the new gray (5) cells.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with an extended gray shape.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def find_original_L():
        width = next(i for i, val in enumerate(output_grid.values[0]) if val != 5)
        height = next(i for i, row in enumerate(output_grid.values) if row[0] != 5)
        return width, height
    
    def calculate_extension_width(grid_width, original_width):
        return round((grid_width * 2/3) - original_width)
    
    def extend_and_fill(orig_width, orig_height, ext_width):
        for col in range(orig_width, orig_width + ext_width):
            top = orig_height - 1 if col < orig_width else 0
            for row in range(top, rows):
                output_grid.set_cell(row, col, 5)
    
    # Find original L
    orig_width, orig_height = find_original_L()
    
    # Calculate extension width
    ext_width = calculate_extension_width(cols, orig_width)
    
    # Extend and fill
    extend_and_fill(orig_width, orig_height, ext_width)
    
    return output_grid
