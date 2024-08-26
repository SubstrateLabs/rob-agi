from rob_agi.colored_grid import ColoredGrid

def solve_639f5a19(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms sky blue (color 8) regions in the input grid into a pattern of 2x2 colored squares.
    
    The solution works as follows:
    1. Find all connected regions of sky blue (color 8) in the input grid.
    2. For each region, apply a repeating pattern of 2x2 colored squares:
       - Even row, even column: [(6, 1), (2, 3)]
       - Odd row or odd column: [(1, 4), (3, 4)]
    3. The pattern is applied consistently across all regions, handling any size including odd dimensions.
    4. Non-sky blue areas in the grid are preserved.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with sky blue regions replaced by the color pattern.
    """
    new_grid = input_grid.deep_copy()
    
    color_pattern = [
        [(6, 1), (2, 3)],  # Even row, even column
        [(1, 4), (3, 4)]   # Odd row or odd column
    ]
    
    sky_blue_regions = input_grid.find_connected_regions(8)
    
    for region in sky_blue_regions:
        min_row = min(r for r, _ in region)
        max_row = max(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_col = max(c for _, c in region)
        
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        for i in range(0, height - 1, 2):
            for j in range(0, width - 1, 2):
                pattern_row = (i // 2) % 2
                pattern_col = (j // 2) % 2
                
                for di in range(2):
                    for dj in range(2):
                        if i + di < height and j + dj < width:
                            new_color = color_pattern[pattern_row][pattern_col][di * 2 + dj]
                            new_grid.values[min_row + i + di][min_col + j + dj] = new_color
    
    return new_grid
