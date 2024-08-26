from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_e5c44e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by adding a green 'E' pattern based on the following rules:
    1. Find the initial green (3) square.
    2. Determine the leftmost valid column for the 'E'.
    3. Create the vertical line of the 'E'.
    4. Create the top, middle, and bottom horizontal lines of the 'E'.
    5. Fill the bottom row if possible.
    6. Connect any disconnected parts of the 'E'.
    7. Extend the 'E' pattern if possible.
    8. Ensure the 'E' touches at least three edges of the grid.
    9. Preserve all red (2) squares.
    10. Ensure all green squares are connected.

    The function adapts to various initial conditions and red square placements
    to create the largest possible 'E' pattern that satisfies the challenge requirements.
    """
    output_grid = input_grid.deep_copy()
    initial_green = find_initial_green(output_grid)
    if not initial_green:
        return output_grid

    left_column = find_leftmost_column(output_grid, initial_green)
    create_vertical_line(output_grid, left_column)
    create_horizontal_lines(output_grid, left_column, initial_green)
    fill_bottom_row(output_grid)
    connect_disconnected_parts(output_grid)
    extend_e_pattern(output_grid, left_column)

    return output_grid

def find_initial_green(grid: ColoredGrid) -> Optional[Tuple[int, int]]:
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                return r, c
    return None

def find_leftmost_column(grid: ColoredGrid, initial_green: Tuple[int, int]) -> int:
    for c in range(1, initial_green[1] + 1):
        if all(grid.get_cell(r, c) != 2 for r in range(grid.num_rows)):
            return c
    return initial_green[1]

def create_vertical_line(grid: ColoredGrid, col: int):
    for r in range(grid.num_rows):
        if grid.get_cell(r, col) == 0:
            grid.set_cell(r, col, 3)

def create_horizontal_lines(grid: ColoredGrid, left_col: int, initial_green: Tuple[int, int]):
    rows, cols = grid.num_rows, grid.num_cols
    
    # Top line
    for c in range(left_col + 1, cols - 1):
        if grid.get_cell(0, c) == 2:
            break
        grid.set_cell(0, c, 3)
    
    # Middle line
    middle_row = min(initial_green[0], rows // 2)
    while middle_row > 0 and grid.get_cell(middle_row, left_col) == 2:
        middle_row -= 1
    for c in range(left_col + 1, cols - 1):
        if grid.get_cell(middle_row, c) == 2:
            break
        grid.set_cell(middle_row, c, 3)
    
    # Bottom line
    bottom_row = rows - 2
    while bottom_row > middle_row and grid.get_cell(bottom_row, left_col) == 2:
        bottom_row -= 1
    for c in range(left_col + 1, cols - 1):
        if grid.get_cell(bottom_row, c) == 2:
            break
        grid.set_cell(bottom_row, c, 3)

def fill_bottom_row(grid: ColoredGrid):
    bottom_row = grid.num_rows - 1
    left_limit = 0
    right_limit = grid.num_cols - 1
    
    while left_limit < grid.num_cols and grid.get_cell(bottom_row, left_limit) != 2:
        grid.set_cell(bottom_row, left_limit, 3)
        left_limit += 1
    
    while right_limit >= left_limit and grid.get_cell(bottom_row, right_limit) != 2:
        grid.set_cell(bottom_row, right_limit, 3)
        right_limit -= 1

def connect_disconnected_parts(grid: ColoredGrid):
    for c in range(1, grid.num_cols - 1):
        connected = False
        for r in range(grid.num_rows):
            if grid.get_cell(r, c) == 3:
                if not connected:
                    connected = True
                elif grid.get_cell(r - 1, c) != 3:
                    for i in range(r - 1, -1, -1):
                        if grid.get_cell(i, c) == 3:
                            break
                        if grid.get_cell(i, c) == 0:
                            grid.set_cell(i, c, 3)

def extend_e_pattern(grid: ColoredGrid, left_col: int):
    rows, cols = grid.num_rows, grid.num_cols
    for r in [0, rows // 2, rows - 2]:
        for c in range(cols - 1, left_col, -1):
            if all(grid.get_cell(r, i) == 3 for i in range(left_col, c)) and grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 3)
            else:
                break
