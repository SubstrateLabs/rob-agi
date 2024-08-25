from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_12997ef3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying unique colors and creating a pattern for each.
    
    The function scans the input grid for unique colors (excluding black),
    sorts them, and creates a new grid where each color is represented by a 3x3 pattern.
    The pattern for each color has the color in the corners and center, with black in between.
    The orientation (horizontal or vertical) of the output grid is determined by the number
    of unique colors found and the dimensions of the input grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    colors_to_process = [2, 3, 4, 5, 6, 7, 8, 9]
    found_colors = []
    
    # Scan the input grid
    for row in input_grid.values:
        for cell in row:
            if cell in colors_to_process and cell not in found_colors:
                found_colors.append(cell)
    
    # Sort the found colors
    found_colors.sort(key=lambda x: colors_to_process.index(x))
    
    # Determine output orientation
    if len(found_colors) <= 3 or input_grid.num_cols > input_grid.num_rows:
        orientation = 'horizontal'
        width, height = 9, 3
    else:
        orientation = 'vertical'
        width, height = 3, 9
    
    # Create the output grid
    output_values = [[0 for _ in range(width)] for _ in range(height)]
    
    for i, color in enumerate(found_colors):
        if orientation == 'horizontal':
            start_row, start_col = 0, i * 3
        else:
            start_row, start_col = i * 3, 0
        
        # Create 3x3 pattern
        for r in range(3):
            for c in range(3):
                if (r, c) in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
                    output_values[start_row + r][start_col + c] = color
    
    return ColoredGrid(values=output_values)
