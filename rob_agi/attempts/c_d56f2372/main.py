from rob_agi.colored_grid import ColoredGrid

def solve_d56f2372(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by extracting the largest connected region of the highest non-zero color value.
    
    1. Finds the highest non-zero color value in the input grid.
    2. Identifies all connected regions of this color.
    3. Selects the largest region.
    4. Determines the bounding box of the selected region.
    5. Creates a new grid with the dimensions of the bounding box.
    6. Copies the selected shape to the new grid, preserving its relative position.
    7. Fills remaining cells with black (0).
    
    Args:
        input_grid (ColoredGrid): The input grid to process.
    
    Returns:
        ColoredGrid: A new grid containing only the extracted shape.
    """
    # Find highest non-zero color value
    max_color = max(cell for row in input_grid.values for cell in row if cell != 0)
    
    # Find regions of highest color
    regions = input_grid.find_connected_regions(max_color)
    
    # Select largest region
    largest_region = max(regions, key=len)
    
    # Determine bounding box
    min_x = min(y for x, y in largest_region)
    max_x = max(y for x, y in largest_region)
    min_y = min(x for x, y in largest_region)
    max_y = max(x for x, y in largest_region)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    # Create new grid
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Copy shape to new grid
    for x, y in largest_region:
        new_grid.values[x - min_y][y - min_x] = max_color
    
    return new_grid
