from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4e469f39(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by identifying gray (5) shapes and adding a red (2) outline.
    
    The function performs the following steps:
    1. Identify all gray shapes in the input grid.
    2. Determine the overall bounding box for all gray shapes.
    3. Create a continuous top red line above all shapes, spanning only the width of the shapes.
    4. Draw vertical red lines on both sides of each shape.
    5. Fill the insides of all gray regions with red, preserving the gray outline.
    
    Args:
    input_grid (ColoredGrid): The input grid containing gray shapes.
    
    Returns:
    ColoredGrid: A new grid with a red outline added around all gray shapes.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    gray_regions = input_grid.find_connected_regions(5)
    
    if not gray_regions:
        return output_grid
    
    # Determine overall bounding box
    top = min(min(row for row, _ in region) for region in gray_regions)
    left = min(min(col for _, col in region) for region in gray_regions)
    right = max(max(col for _, col in region) for region in gray_regions)
    
    # Create continuous top red line
    for col in range(left, right + 1):
        output_grid.values[top-1][col] = 2
    
    # Process each region
    for region in gray_regions:
        region_left = min(col for _, col in region)
        region_right = max(col for _, col in region)
        region_top = min(row for row, _ in region)
        region_bottom = max(row for row, _ in region)
        
        # Draw vertical lines
        for row in range(top-1, region_bottom+1):
            if output_grid.values[row][region_left-1] == 0:
                output_grid.values[row][region_left-1] = 2
            if output_grid.values[row][region_right+1] == 0:
                output_grid.values[row][region_right+1] = 2
        
        # Fill inside of region
        for row in range(region_top, region_bottom+1):
            for col in range(region_left, region_right+1):
                if input_grid.values[row][col] == 0:
                    output_grid.values[row][col] = 2
    
    return output_grid
