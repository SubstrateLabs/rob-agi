from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4e469f39(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by identifying gray (5) shapes and adding red (2) outlines.
    
    The function performs the following steps:
    1. Identify all gray shapes in the input grid.
    2. For each shape, determine its orientation based on the grid's midpoint.
    3. Draw a red horizontal "roof" line above each shape, extending to the grid edge on one side.
    4. Draw a red vertical "wall" line from the shape to the top of the grid on the appropriate side.
    5. Complete the shape outline by filling gaps between gray cells with red.
    
    Args:
    input_grid (ColoredGrid): The input grid containing gray shapes.
    
    Returns:
    ColoredGrid: A new grid with red outlines added around the gray shapes.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    grid_midpoint = cols // 2
    gray_regions = input_grid.find_connected_regions(5)
    
    for region in gray_regions:
        top = min(row for row, _ in region)
        left = min(col for _, col in region)
        right = max(col for _, col in region)
        shape_midpoint = (left + right) // 2
        
        # Draw horizontal "roof" line
        if shape_midpoint < grid_midpoint:
            for col in range(left):
                output_grid.values[top-1][col] = 2
        else:
            for col in range(right + 1, cols):
                output_grid.values[top-1][col] = 2
        
        # Determine orientation and draw vertical "wall" line
        if shape_midpoint < grid_midpoint:
            wall_col = left
            for row in range(top-1, -1, -1):
                if output_grid.values[row][wall_col] == 0:
                    output_grid.values[row][wall_col] = 2
                else:
                    break
        else:
            wall_col = right
            for row in range(top-1, -1, -1):
                if output_grid.values[row][wall_col] == 0:
                    output_grid.values[row][wall_col] = 2
                else:
                    break
        
        # Complete shape outline
        for col in range(left, right):
            if output_grid.values[top][col] == 0 and output_grid.values[top][col+1] == 5:
                output_grid.values[top][col] = 2
        
        if shape_midpoint < grid_midpoint:
            for row in range(top, rows):
                if output_grid.values[row][left] == 0 and row+1 < rows and output_grid.values[row+1][left] == 5:
                    output_grid.values[row][left] = 2
        else:
            for row in range(top, rows):
                if output_grid.values[row][right] == 0 and row+1 < rows and output_grid.values[row+1][right] == 5:
                    output_grid.values[row][right] = 2
    
    return output_grid
