from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_f5c89df1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by identifying the core shape formed by sky blue (8) squares,
    and creating an idealized, symmetrical version of it.
    
    1. Analyze the input shape to determine its key characteristics.
    2. Design a new symmetrical shape based on the input's structure.
    3. Size and position the new shape within the grid.
    4. Create the output grid with the new shape.
    
    The transformation aims to preserve the general structure and proportions
    of the input while creating a more symmetrical and centered output.
    
    Returns a new grid with the transformed pattern.
    """
    # Step 1: Analyze input shape
    sky_blue_regions = input_grid.find_connected_regions(8)
    if not sky_blue_regions:
        return input_grid  # No transformation needed
    
    largest_region = max(sky_blue_regions, key=len)
    min_r = min(r for r, _ in largest_region)
    max_r = max(r for r, _ in largest_region)
    min_c = min(c for _, c in largest_region)
    max_c = max(c for _, c in largest_region)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    # Step 2: Design new shape
    new_height = min(max(height, 5), 9)
    new_width = min(max(width, 5), 9)
    
    new_shape = set()
    for r in range(new_height):
        for c in range(new_width):
            if (r == 0 or r == new_height - 1 or c == 0 or c == new_width - 1 or
                r == new_height // 2 or c == new_width // 2):
                new_shape.add((r, c))
    
    # Step 3: Position new shape
    top_row = 2
    left_col = (13 - new_width) // 2
    
    # Step 4: Create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(13)] for _ in range(13)])
    for r, c in new_shape:
        output_grid.set_cell(top_row + r, left_col + c, 8)
    
    return output_grid
