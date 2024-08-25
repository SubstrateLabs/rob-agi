from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4e469f39(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by identifying gray (5) 'P' shapes and adding red (2) outlines.
    
    The function performs the following steps:
    1. Identify all gray shapes in the input grid.
    2. For each shape, determine its orientation and critical corner.
    3. Draw a red outline around the top and outer edge of each gray shape.
    4. The red outline extends to the edge of the grid horizontally and to the top of the grid vertically.
    
    Args:
    input_grid (ColoredGrid): The input grid containing gray shapes.
    
    Returns:
    ColoredGrid: A new grid with red outlines added around the gray shapes.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    gray_regions = input_grid.find_connected_regions(5)
    
    for region in gray_regions:
        # Determine orientation and critical corner
        left_most = min(col for _, col in region)
        right_most = max(col for _, col in region)
        top_most = min(row for row, _ in region)
        
        if left_most < cols - right_most:
            # Shape is closer to left edge
            critical_corner = (top_most, left_most)
            direction = -1  # Move left
        else:
            # Shape is closer to right edge
            critical_corner = (top_most, right_most)
            direction = 1  # Move right
        
        # Draw horizontal line
        row = critical_corner[0] - 1
        col = critical_corner[1]
        while 0 <= col < cols:
            if output_grid.values[row][col] == 0:
                output_grid.values[row][col] = 2
            col += direction
        
        # Draw vertical line
        row = critical_corner[0]
        col = critical_corner[1] + direction
        while row >= 0:
            if output_grid.values[row][col] == 0:
                output_grid.values[row][col] = 2
            row -= 1
    
    return output_grid
