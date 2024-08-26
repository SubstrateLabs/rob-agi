from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_29700607(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing lines connecting colored squares.
    
    For each color:
    1. Determines if the line should be horizontal or vertical based on the positions of the colored squares.
    2. For horizontal lines, draws in the row with the most instances of that color, from leftmost to rightmost occurrence.
    3. For vertical lines, draws in the column with the most instances of that color, from topmost to bottommost occurrence.
    
    Returns a new ColoredGrid with the drawn lines.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    color_positions = get_color_positions(input_grid)
    
    for color, positions in color_positions.items():
        direction = determine_direction(positions)
        if direction == 'horizontal':
            draw_horizontal_line(color, positions, output_grid)
        else:
            draw_vertical_line(color, positions, output_grid)
    
    return output_grid

def get_color_positions(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    color_positions = {}
    rows, cols = grid.get_dimensions()
    for row in range(rows):
        for col in range(cols):
            color = grid.values[row][col]
            if color != 0:
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((row, col))
    return color_positions

def determine_direction(positions: List[Tuple[int, int]]) -> str:
    rows = set(pos[0] for pos in positions)
    cols = set(pos[1] for pos in positions)
    return 'horizontal' if len(rows) < len(cols) else 'vertical'

def draw_horizontal_line(color: int, positions: List[Tuple[int, int]], output_grid: ColoredGrid):
    row = max(set(pos[0] for pos in positions), key=lambda r: sum(1 for p in positions if p[0] == r))
    left = min(pos[1] for pos in positions if pos[0] == row)
    right = max(pos[1] for pos in positions if pos[0] == row)
    for col in range(left, right + 1):
        output_grid.values[row][col] = color

def draw_vertical_line(color: int, positions: List[Tuple[int, int]], output_grid: ColoredGrid):
    col = max(set(pos[1] for pos in positions), key=lambda c: sum(1 for p in positions if p[1] == c))
    top = min(pos[0] for pos in positions if pos[1] == col)
    bottom = max(pos[0] for pos in positions if pos[1] == col)
    for row in range(top, bottom + 1):
        output_grid.values[row][col] = color
