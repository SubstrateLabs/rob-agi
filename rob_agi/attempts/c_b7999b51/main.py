from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b7999b51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing non-black color regions into columns.
    
    The function performs the following steps:
    1. Identifies all distinct non-black colors and their positions.
    2. Sorts colors based on their bottom-most and rightmost positions.
    3. Creates a new grid with columns representing each color, preserving their relative heights.
    4. Optimizes the output by removing any completely black rows from the top.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with compressed color columns.
    """
    # Step 1: Analyze the input grid
    colors_info = []
    max_height = 0
    for color in range(1, 10):  # Exclude black (0)
        regions = input_grid.find_connected_regions(color)
        if regions:
            bottom = max(r for region in regions for r, _ in region)
            right = max(c for region in regions for _, c in region)
            height = max(len(region) for region in regions)
            colors_info.append((color, bottom, right, height))
            max_height = max(max_height, height)
    
    # Step 2: Sort colors
    colors_info.sort(key=lambda x: (-x[1], -x[2]))
    
    # Step 3 & 4: Create and fill the output grid
    output_width = len(colors_info)
    output_grid = ColoredGrid(values=[[0 for _ in range(output_width)] for _ in range(max_height)])
    
    for col, (color, _, _, height) in enumerate(colors_info):
        for row in range(max_height - height, max_height):
            output_grid.values[row][col] = color
    
    # Step 5: Optimize the output (remove black rows from the top)
    first_non_black_row = 0
    while first_non_black_row < max_height and all(cell == 0 for cell in output_grid.values[first_non_black_row]):
        first_non_black_row += 1
    
    if first_non_black_row > 0:
        output_grid = ColoredGrid(values=output_grid.values[first_non_black_row:])
    
    return output_grid
