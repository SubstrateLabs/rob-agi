from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4b6b68e5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling closed regions with their highest-valued color.
    
    The solution follows these steps:
    1. Identify all closed regions in the grid for each non-zero color.
    2. For each region, find the highest-valued color within it (including the boundary).
    3. If the highest color is different from the boundary, fill the entire region with that color.
    4. If the highest color is the same as the boundary, leave the region unchanged.
    5. Return the modified grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid after applying the filling rules.
    """
    output_grid = input_grid.deep_copy()
    colors = set(color for row in input_grid.values for color in row if color != 0)
    
    for color in colors:
        regions = input_grid.find_connected_regions(color)
        for region in regions:
            highest_color = max(max(input_grid.get_cell(r, c) for r, c in region), color)
            if highest_color != color:
                for r, c in region:
                    output_grid.set_cell(r, c, highest_color)
    
    return output_grid
