from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_12997ef3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying unique colors and creating a pattern for each.
    
    The function scans the input grid for unique colors (excluding black and blue),
    preserves their order of appearance, and creates a new grid where each color is
    represented by a 3x3 pattern. The pattern for each color has the color in the center
    and four corners, with black in between. The orientation (horizontal or vertical)
    of the output grid is determined by the spatial relationship of colors in the input grid.
    
    The 3x3 pattern for each color is as follows:
    [color, 0, color]
    [0, color, 0]
    [color, 0, color]
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    def scan_input_grid(grid: ColoredGrid) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
        colors = []
        color_positions = {}
        for row in range(grid.num_rows):
            for col in range(grid.num_cols):
                color = grid.values[row][col]
                if color > 1 and color not in colors:
                    colors.append(color)
                    color_positions[color] = (row, col)
        return colors, color_positions

    def determine_orientation(color_positions: Dict[int, Tuple[int, int]]) -> str:
        if len(color_positions) <= 1:
            return 'vertical'  # Default to vertical if only one or no color
        
        positions = list(color_positions.values())
        vertical_distances = [abs(positions[i+1][0] - positions[i][0]) for i in range(len(positions)-1)]
        horizontal_distances = [abs(positions[i+1][1] - positions[i][1]) for i in range(len(positions)-1)]
        
        avg_vertical = sum(vertical_distances) / len(vertical_distances) if vertical_distances else 0
        avg_horizontal = sum(horizontal_distances) / len(horizontal_distances) if horizontal_distances else 0
        
        return 'vertical' if avg_vertical >= avg_horizontal else 'horizontal'

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
                [color, 0, color]
            ]
        
            for r in range(3):
                for c in range(3):
                    output[start_row + r][start_col + c] = pattern[r][c]
    
        return output

    colors, color_positions = scan_input_grid(input_grid)
    orientation = determine_orientation(color_positions)
    output_values = create_output_grid(colors, orientation)
    return ColoredGrid(values=output_values)
