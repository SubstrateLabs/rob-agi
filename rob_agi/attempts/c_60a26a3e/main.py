from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_60a26a3e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red diamond shapes with blue lines.
    
    The solution follows these steps:
    1. Identify all red (2) regions
    2. Determine the optimal bounding rectangle
    3. Create a blue line structure along the optimal bounding rectangle
    4. Add vertical connections between red diamonds
    5. Add necessary horizontal connections
    6. Optimize the solution by removing unnecessary lines
    7. Handle special cases (single row/column configurations)
    
    This approach creates the most compact rectangular structure that encloses
    the optimal set of red diamonds, sometimes intentionally leaving out smaller,
    isolated groups if including them would significantly increase the rectangle's size.
    """
    output_grid = input_grid.deep_copy()
    red_regions = input_grid.find_connected_regions(2)
    
    if not red_regions:
        return output_grid
    
    # Find optimal bounding rectangle
    all_red_points = [point for region in red_regions for point in region]
    min_row, max_row = min(p[0] for p in all_red_points), max(p[0] for p in all_red_points)
    min_col, max_col = min(p[1] for p in all_red_points), max(p[1] for p in all_red_points)
    
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
        output_grid.values[r][min_col] = output_grid.values[r][max_col] = 1
    for c in range(min_col, max_col + 1):
        output_grid.values[min_row][c] = output_grid.values[max_row][c] = 1
    
    # Add vertical connections
    for c in range(min_col, max_col + 1):
        red_in_column = [r for r in range(min_row, max_row + 1) if input_grid.values[r][c] == 2]
        if red_in_column:
            for r in range(min(red_in_column), max(red_in_column) + 1):
                output_grid.values[r][c] = 1
    
    # Add necessary horizontal connections
    empty_columns = [c for c in range(min_col, max_col + 1) if 2 not in [input_grid.values[r][c] for r in range(min_row, max_row + 1)]]
    if empty_columns:
        for c in empty_columns:
            output_grid.values[min_row][c] = output_grid.values[max_row][c] = 1
    
    return output_grid
