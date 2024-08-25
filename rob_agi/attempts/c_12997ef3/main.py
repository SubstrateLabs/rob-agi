from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_12997ef3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying unique colors and creating a pattern for each.
    
    The function scans the input grid for the most prominent line of unique colors (excluding black and blue),
    determines the orientation (horizontal or vertical) based on this line, and creates a new grid where each color
    is represented by a 3x3 pattern. The pattern for each color has the color in the center and four corners,
    with black in between, except for the bottom row which is filled with the color.
    
    The 3x3 pattern for each color is as follows:
    [color, 0, color]
    [0, color, 0]
    [color, color, color]
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    def scan_input_grid(grid: ColoredGrid) -> Tuple[List[int], str]:
        horizontal_colors = []
        vertical_colors = []
        for row in range(grid.num_rows):
            row_colors = [color for color in set(grid.values[row]) if color > 1]
            if len(row_colors) > len(horizontal_colors):
                horizontal_colors = row_colors
        for col in range(grid.num_cols):
            col_colors = [color for color in set(grid.values[r][col] for r in range(grid.num_rows)) if color > 1]
            if len(col_colors) > len(vertical_colors):
                vertical_colors = col_colors
        
        if len(horizontal_colors) >= len(vertical_colors):
            return horizontal_colors, 'horizontal'
        else:
            return vertical_colors, 'vertical'

    def create_output_grid(colors: List[int], orientation: str) -> List[List[int]]:
        if orientation == 'horizontal':
            width, height = len(colors) * 3, 3
        else:
            width, height = 3, len(colors) * 3
    
        output = [[0 for _ in range(width)] for _ in range(height)]
    
        for i, color in enumerate(colors):
            if orientation == 'horizontal':
                start_row, start_col = 0, i * 3
            else:
                start_row, start_col = i * 3, 0
        
            pattern = [
                [color, 0, color],
                [0, color, 0],
                [color, color, color]
            ]
        
            for r in range(3):
                for c in range(3):
                    output[start_row + r][start_col + c] = pattern[r][c]
    
        return output

    colors, orientation = scan_input_grid(input_grid)
    output_values = create_output_grid(colors, orientation)
    return ColoredGrid(values=output_values)
