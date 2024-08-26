from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_29700607(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing lines connecting colored squares.
    
    For each color:
    1. Determines if the line should be horizontal or vertical based on the spread of colored squares.
    2. For horizontal lines, draws in the topmost row where the color appears, from leftmost to rightmost occurrence.
    3. For vertical lines, draws in the leftmost column where the color appears, from topmost to bottommost occurrence.
    
    Returns a new ColoredGrid with the drawn lines.
    """
    output_grid = input_grid.deep_copy()
    color_positions = get_color_positions(input_grid)
    
    for color, positions in color_positions.items():
        direction = determine_line_direction(positions)
        if direction == 'horizontal':
            draw_horizontal_line(output_grid, color, positions)
        else:
            draw_vertical_line(output_grid, color, positions)
    
    return output_grid

def get_color_positions(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    color_positions = {}
    for row in range(len(grid.values)):
        for col in range(len(grid.values[0])):
            color = grid.values[row][col]
            if color != 0:
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((row, col))
    return color_positions

def determine_line_direction(positions: List[Tuple[int, int]]) -> str:
    rows, cols = zip(*positions)
    horizontal_spread = max(cols) - min(cols)
    vertical_spread = max(rows) - min(rows)
    return 'horizontal' if horizontal_spread >= vertical_spread else 'vertical'

def draw_horizontal_line(grid: ColoredGrid, color: int, positions: List[Tuple[int, int]]):
    rows, cols = zip(*positions)
    top_row = min(rows)
    left_col, right_col = min(cols), max(cols)
    for col in range(left_col, right_col + 1):
        grid.values[top_row][col] = color

def draw_vertical_line(grid: ColoredGrid, color: int, positions: List[Tuple[int, int]]):
    rows, cols = zip(*positions)
    left_col = min(cols)
    top_row, bottom_row = min(rows), max(rows)
    for row in range(top_row, bottom_row + 1):
        grid.values[row][left_col] = color
