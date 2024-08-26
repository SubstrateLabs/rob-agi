from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9c56f360(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving individual green (3) squares as far left and up as possible
    while maintaining contact with at least one sky blue (8) square (including diagonally)
    and avoiding overlap with other green squares.

    1. Identifies all green squares in the grid.
    2. Processes each green square from left to right, top to bottom.
    3. For each green square, finds the best new position (leftmost, then topmost).
    4. Moves the green square to the new position if different from the original.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with green squares moved.
    """
    grid = input_grid.deep_copy()
    green_squares = find_green_squares(grid)
    green_squares.sort(key=lambda x: (x[1], x[0]))  # Sort by column, then row
    
    for row, col in green_squares:
        new_row, new_col = find_best_position(grid, row, col)
        if (new_row, new_col) != (row, col):
            grid.set_cell(row, col, 0)  # Set original position to black
            grid.set_cell(new_row, new_col, 3)  # Set new position to green
    
    return grid

def find_green_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
    green_squares = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                green_squares.append((r, c))
    return green_squares

def is_valid_position(grid: ColoredGrid, row: int, col: int) -> bool:
    rows, cols = grid.get_dimensions()
    return 0 <= row < rows and 0 <= col < cols

def is_adjacent_to_sky_blue(grid: ColoredGrid, row: int, col: int) -> bool:
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if is_valid_position(grid, nr, nc) and grid.get_cell(nr, nc) == 8:
                return True
    return False

def find_best_position(grid: ColoredGrid, start_row: int, start_col: int) -> Tuple[int, int]:
    best_row, best_col = start_row, start_col
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if (c < best_col or (c == best_col and r < best_row)) and \
               is_valid_position(grid, r, c) and \
               grid.get_cell(r, c) != 3 and \
               is_adjacent_to_sky_blue(grid, r, c):
                best_row, best_col = r, c
    
    return best_row, best_col
