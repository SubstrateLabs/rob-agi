from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_9c56f360(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving green (3) squares as far left and up as possible
    while maintaining contact with at least one sky blue (8) square (including diagonally)
    and avoiding overlap with other green squares.

    1. Creates a deep copy of the input grid.
    2. Identifies all green squares in the grid.
    3. Sorts green squares from left to right, then top to bottom.
    4. For each green square or connected group:
       a. Attempts to move left as far as possible.
       b. If can't move left, attempts to move up as far as possible.
       c. If can't move individually, considers it as part of a group.
    5. Repeats the process until no more moves are possible.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with green squares moved.
    """
    grid = input_grid.deep_copy()
    changed = True
    while changed:
        changed = False
        green_squares = find_green_squares(grid)
        green_squares.sort(key=lambda pos: (pos[1], pos[0]))  # Sort left to right, then top to bottom
        
        for pos in green_squares:
            if move_square(grid, pos):
                changed = True
    
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

def has_sky_blue_neighbor(grid: ColoredGrid, row: int, col: int) -> bool:
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if is_valid_position(grid, nr, nc) and grid.get_cell(nr, nc) == 8:
                return True
    return False

def move_square(grid: ColoredGrid, pos: Tuple[int, int]) -> bool:
    row, col = pos
    # Try to move left
    for new_col in range(col - 1, -1, -1):
        if can_move_to(grid, row, new_col):
            grid.set_cell(row, col, 0)
            grid.set_cell(row, new_col, 3)
            return True
    
    # Try to move up
    for new_row in range(row - 1, -1, -1):
        if can_move_to(grid, new_row, col):
            grid.set_cell(row, col, 0)
            grid.set_cell(new_row, col, 3)
            return True
    
    return False

def can_move_to(grid: ColoredGrid, row: int, col: int) -> bool:
    return (is_valid_position(grid, row, col) and
            grid.get_cell(row, col) == 0 and
            has_sky_blue_neighbor(grid, row, col))
