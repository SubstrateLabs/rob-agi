from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_12997ef3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying unique colors and creating a pattern for each.
    
    The function scans the input grid for all unique colors (excluding black),
    determines the orientation (horizontal or vertical) based on the number of unique colors,
    and creates a new grid where each color is represented by a 3x3 pattern.
    The pattern for each color has the color in the top-right, bottom-left, and center,
    with black in the other positions.
    
    The 3x3 pattern for each color is as follows:
    [0, color, color]
    [color, color, 0]
    [0, color, color]
    
    The colors are ordered based on their first appearance in the input grid,
    scanning from top-left to bottom-right.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    def scan_input_grid(grid: ColoredGrid) -> List[int]:
        color_positions = {}
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                color = grid.values[r][c]
                if color != 0 and color not in color_positions:
                    color_positions[color] = (r, c)
        return sorted(color_positions.keys(), key=lambda x: color_positions[x])

    def create_output_grid(colors: List[int]) -> List[List[int]]:
        orientation = 'vertical' if len(colors) >= 3 else 'horizontal'
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
                [0, color, color],
                [color, color, 0],
                [0, color, color]
            ]
        
            for r in range(3):
                for c in range(3):
                    output[start_row + r][start_col + c] = pattern[r][c]
    
        return output

    colors = scan_input_grid(input_grid)
    output_values = create_output_grid(colors)
    return ColoredGrid(values=output_values)
