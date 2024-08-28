from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e2092e0c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by extending the gray 'L' shape in the top-left corner.
    
    The solution follows these steps:
    1. Analyze the input grid to find the dimensions and the existing gray 'L' shape.
    2. Calculate the target dimensions based on 2/3 of the grid size.
    3. Determine the extension dimensions, ensuring to meet or exceed the target area.
    4. Create a new grid as a deep copy of the input grid.
    5. Fill the extension area with gray (5), overwriting any existing colors.
    6. Ensure the extension doesn't go beyond grid boundaries.
    
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
                width = max(width, i + 1)
            if grid.get_cell(i, 0) == 5:
                height = max(height, i + 1)
        return width, height
    
    # Find original L
    orig_width, orig_height = find_gray_L(output_grid)
    
    # Calculate target and extension dimensions
    target_width = round(cols * 2/3)
    target_height = round(rows * 2/3)
    ext_width = max(target_width, orig_width)
    ext_height = max(target_height, orig_height)
    
    # Adjust dimensions to meet target area if necessary
    target_area = target_width * target_height
    while ext_width * ext_height < target_area:
        if ext_width < ext_height:
            ext_width += 1
        else:
            ext_height += 1
    
    # Ensure extension doesn't exceed grid dimensions
    ext_width = min(ext_width, cols)
    ext_height = min(ext_height, rows)
    
    # Fill the extension area
    for row in range(ext_height):
        for col in range(ext_width):
            output_grid.set_cell(row, col, 5)
    
    return output_grid
