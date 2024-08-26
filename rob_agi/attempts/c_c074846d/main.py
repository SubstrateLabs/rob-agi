from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List
import copy

def solve_c074846d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid according to the following rules:
    1. Finds the gray (5) square and the two nearest red (2) squares.
    2. Changes the red squares to green (3).
    3. Adds new red squares perpendicular to the original red line,
       aligned with the end furthest from the gray square.
    4. Ensures all transformations stay within the grid boundaries.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()
    
    # Find gray square
    gray_pos = next((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5)
    
    # Find nearest red squares
    red_squares = find_nearest_red_squares(input_grid, gray_pos)
    
    # Change red to green
    for r, c in red_squares:
        new_grid.values[r][c] = 3
    
    # Determine orientation and new red square positions
    if red_squares[0][0] == red_squares[1][0]:  # Horizontal
        new_red_col = max(s[1] for s in red_squares)
        new_red_squares = [(gray_pos[0] - 1, new_red_col), (gray_pos[0] - 2, new_red_col)]
    else:  # Vertical
        new_red_row = max(s[0] for s in red_squares)
        new_red_squares = [(new_red_row, gray_pos[1] - 1), (new_red_row, gray_pos[1] - 2)]
    
    # Place new red squares within bounds
    for r, c in new_red_squares:
        if 0 <= r < rows and 0 <= c < cols:
            new_grid.values[r][c] = 2
    
    return new_grid

def find_nearest_red_squares(grid: ColoredGrid, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    queue = [start]
    visited = set([start])
    red_squares = []
    
    while queue and len(red_squares) < 2:
        r, c = queue.pop(0)
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                visited.add((nr, nc))
                if grid.values[nr][nc] == 2:
                    red_squares.append((nr, nc))
                queue.append((nr, nc))
    
    return red_squares
