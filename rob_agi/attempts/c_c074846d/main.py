from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_c074846d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid according to the following rules:
    1. Finds the gray (5) square and the line of red (2) squares.
    2. Changes the red squares to green (3).
    3. Adds new red squares perpendicular to the original red line,
       extending away from the gray square, with the same length as the original line.
    4. Ensures all transformations stay within the grid boundaries.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()
    
    # Find gray square
    gray_pos = next((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5)
    
    # Find red line
    red_line = find_red_line(input_grid, gray_pos)
    
    # Change red to green
    for r, c in red_line:
        new_grid.values[r][c] = 3
    
    # Determine orientation and new red square positions
    if len(red_line) > 1:
        if red_line[0][0] == red_line[1][0]:  # Horizontal
            new_red_squares = [(gray_pos[0] - i - 1, gray_pos[1]) for i in range(len(red_line))]
        else:  # Vertical
            new_red_squares = [(gray_pos[0], gray_pos[1] + i + 1) for i in range(len(red_line))]
    else:  # Single red square
        if red_line[0][0] == gray_pos[0]:  # Horizontal
            new_red_squares = [(gray_pos[0] - 1, gray_pos[1])]
        else:  # Vertical
            new_red_squares = [(gray_pos[0], gray_pos[1] + 1)]
    
    # Place new red squares within bounds
    for r, c in new_red_squares:
        if 0 <= r < rows and 0 <= c < cols:
            new_grid.values[r][c] = 2
    
    return new_grid

def find_red_line(grid: ColoredGrid, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    red_line = []
    
    for dr, dc in directions:
        line = []
        r, c = start[0] + dr, start[1] + dc
        while 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == 2:
            line.append((r, c))
            r, c = r + dr, c + dc
        if len(line) > len(red_line):
            red_line = line
    
    return red_line
