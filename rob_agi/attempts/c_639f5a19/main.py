from rob_agi.colored_grid import ColoredGrid

def solve_639f5a19(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms sky blue (color 8) regions in the input grid into a repeating 6x6 color pattern.
    
    The solution works as follows:
    1. Find all connected regions of sky blue (color 8) in the input grid.
    2. For each region, apply a repeating 6x6 color pattern:
       [6, 6, 1, 1, 1, 1]
       [6, 6, 1, 1, 1, 1]
       [6, 6, 4, 4, 4, 4]
       [6, 6, 4, 4, 4, 4]
       [2, 2, 4, 4, 4, 4]
       [2, 2, 3, 3, 3, 3]
    3. The pattern is applied consistently across all regions, handling any size and position.
    4. Non-sky blue areas in the grid are preserved.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with sky blue regions replaced by the color pattern.
    """
    new_grid = input_grid.deep_copy()
    pattern = [
        [6, 6, 1, 1, 1, 1],
        [6, 6, 1, 1, 1, 1],
        [6, 6, 4, 4, 4, 4],
        [6, 6, 4, 4, 4, 4],
        [2, 2, 4, 4, 4, 4],
        [2, 2, 3, 3, 3, 3]
    ]
    
    sky_blue_regions = input_grid.find_connected_regions(8)
    
    for region in sky_blue_regions:
        min_row = min(r for r, _ in region)
        max_row = max(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_col = max(c for _, c in region)
        
        start_row = 2 if min_row % 2 == 1 else 0
        start_col = 2 if min_col % 2 == 1 else 0
        
        for cell_row, cell_col in region:
            pattern_row = (start_row + cell_row - min_row) % 6
            pattern_col = (start_col + cell_col - min_col) % 6
            new_color = pattern[pattern_row][pattern_col]
            new_grid.values[cell_row][cell_col] = new_color
    
    return new_grid
