from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b20f7c8b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying specific rules to 5x5 regions.
    
    The transformation rules are:
    1. If a 5x5 region is solid, it's transformed into a specific pattern.
    2. If a 5x5 region has a pattern, it's transformed into a solid color.
    3. The color mapping for the transformation depends on the position and original colors of the region.
    
    The function identifies 5x5 regions in the middle and right side of the grid,
    analyzes them, and applies the appropriate transformation.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Identify and transform 5x5 regions
    for col in range(6, cols - 4, 7):  # Start at column 6 and 16
        for row in range(1, rows - 4, 7):  # Start at row 1 and 9
            region = input_grid.extract_subgrid(row, col, 5, 5)
            if is_solid_region(region):
                new_color = get_new_color_for_solid(region, col)
                fill_region_solid(output_grid, row, col, new_color)
            else:
                new_color = get_new_color_for_pattern(region, col)
                fill_region_solid(output_grid, row, col, new_color)
    
    return output_grid

def is_solid_region(region: ColoredGrid) -> bool:
    """Check if the region is a solid color."""
    return len(set(cell for row in region.values for cell in row)) == 1

def get_new_color_for_solid(region: ColoredGrid, col: int) -> int:
    """Determine the new color for a solid region based on its position."""
    original_color = region.values[0][0]
    if col == 6:  # Left side
        return (original_color + 1) % 10
    else:  # Right side
        return (original_color + 3) % 10

def get_new_color_for_pattern(region: ColoredGrid, col: int) -> int:
    """Determine the new color for a patterned region based on its position."""
    if col == 6:  # Left side
        return 2 if 1 in region.values[0] else 3
    else:  # Right side
        return 4 if 1 in region.values[0] else 5

def fill_region_solid(grid: ColoredGrid, row: int, col: int, color: int):
    """Fill a 5x5 region with a solid color."""
    for i in range(5):
        for j in range(5):
            grid.values[row + i][col + j] = color
