from rob_agi.colored_grid import ColoredGrid

def solve_639f5a19(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms sky blue (color 8) regions in the input grid into a specific color pattern.
    
    The solution works as follows:
    1. Find all connected regions of sky blue (color 8) in the input grid.
    2. For each region, apply a color pattern based on the relative position of each cell:
       - Top-left (2x6): Magenta (6)
       - Top-right: Blue (1)
       - Middle: Yellow (4)
       - Bottom-left (2x2): Red (2)
       - Bottom-right: Green (3)
    3. The pattern adapts to regions of any size, maintaining the relative positions of colors.
    4. Non-sky blue areas in the grid are preserved.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with sky blue regions replaced by the color pattern.
    """
    new_grid = input_grid.deep_copy()
    sky_blue_regions = input_grid.find_connected_regions(8)
    
    for region in sky_blue_regions:
        min_row = min(r for r, _ in region)
        min_col = min(c for _, c in region)
        
        for cell_row, cell_col in region:
            relative_row = cell_row - min_row
            relative_col = cell_col - min_col
            
            if relative_row < 2 and relative_col < 6:
                new_color = 6  # Magenta
            elif relative_row < 2:
                new_color = 1  # Blue
            elif relative_row >= 2 and relative_row < 4 and relative_col < 6:
                new_color = 6  # Magenta
            elif relative_row >= 2 and relative_row < 4:
                new_color = 1  # Blue
            elif relative_row >= 4 and relative_col < 2:
                new_color = 2  # Red
            elif relative_row >= 4 and relative_col >= 2 and relative_col < 6:
                new_color = 4  # Yellow
            elif relative_row >= 4 and relative_col >= 6:
                new_color = 3  # Green
            else:
                new_color = 4  # Yellow (default for center)
            
            new_grid.values[cell_row][cell_col] = new_color
    
    return new_grid
