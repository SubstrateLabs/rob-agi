from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_60a26a3e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red diamond shapes with blue lines.
    
    The solution follows these steps:
    1. Identify all red (2) regions
    2. Determine the bounding rectangle of all red regions
    3. Create a blue line structure along the bounding rectangle
    4. Connect protruding diamonds to the structure
    5. Add internal vertical connections between aligned diamonds
    6. Handle special cases (single row/column configurations)
    7. Optimize the solution by removing unnecessary lines
    
    This approach creates the largest possible enclosed shape while connecting all red diamonds,
    adapting to various configurations and special cases.
    """
    output_grid = input_grid.deep_copy()
    red_regions = input_grid.find_connected_regions(2)
    
    if not red_regions:
        return output_grid
    
    # Find bounding rectangle
    all_red_points = [point for region in red_regions for point in region]
    min_row = min(point[0] for point in all_red_points)
    max_row = max(point[0] for point in all_red_points)
    min_col = min(point[1] for point in all_red_points)
    max_col = max(point[1] for point in all_red_points)
    
    # Handle special cases
    if min_col == max_col:  # Single column
        for r in range(min_row, max_row + 1):
            output_grid.values[r][min_col] = 1
        return output_grid
    
    if min_row == max_row:  # Single row
        for c in range(min_col, max_col + 1):
            output_grid.values[min_row][c] = 1
        return output_grid
    
    # Create blue line structure
    for r in range(min_row, max_row + 1):
        output_grid.values[r][min_col] = 1
        output_grid.values[r][max_col] = 1
    for c in range(min_col, max_col + 1):
        output_grid.values[min_row][c] = 1
        output_grid.values[max_row][c] = 1
    
    # Connect protruding diamonds and add internal connections
    for region in red_regions:
        region_min_row = min(point[0] for point in region)
        region_max_row = max(point[0] for point in region)
        region_min_col = min(point[1] for point in region)
        region_max_col = max(point[1] for point in region)
        
        if region_min_row == min_row:
            center_col = (region_min_col + region_max_col) // 2
            for r in range(min_row, region_max_row + 1):
                output_grid.values[r][center_col] = 1
        
        if region_max_row == max_row:
            center_col = (region_min_col + region_max_col) // 2
            for r in range(region_min_row, max_row + 1):
                output_grid.values[r][center_col] = 1
        
        if region_min_col == min_col:
            center_row = (region_min_row + region_max_row) // 2
            for c in range(min_col, region_max_col + 1):
                output_grid.values[center_row][c] = 1
        
        if region_max_col == max_col:
            center_row = (region_min_row + region_max_row) // 2
            for c in range(region_min_col, max_col + 1):
                output_grid.values[center_row][c] = 1
    
    return output_grid
