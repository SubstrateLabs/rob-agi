from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_e5c44e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by adding a green 'E' pattern based on the following rules:
    1. Find the initial green (3) square.
    2. Determine the leftmost valid column for the 'E'.
    3. Create the vertical line of the 'E'.
    4. Create the top, middle, and bottom horizontal lines of the 'E'.
    5. Ensure the 'E' touches at least three edges of the grid.
    6. Fill the bottom row if possible.
    7. Connect any disconnected parts of the 'E'.
    8. Optimize the 'E' shape.
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
    ensure_three_edge_contact(output_grid, left_column)
    fill_bottom_row(output_grid, left_column)
    connect_disconnected_parts(output_grid)
    optimize_e_shape(output_grid, left_column)

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
    top_row = 0 if grid.get_cell(0, left_col) != 2 else 1
    for c in range(left_col + 1, cols):
        if grid.get_cell(top_row, c) == 2:
            break
        grid.set_cell(top_row, c, 3)
    
    # Middle line
    middle_row = min(initial_green[0], rows // 2)
    while middle_row > 0 and grid.get_cell(middle_row, left_col) == 2:
        middle_row -= 1
    for c in range(left_col + 1, min(cols, left_col + cols // 2)):
        if grid.get_cell(middle_row, c) == 2:
            break
        grid.set_cell(middle_row, c, 3)
    
    # Bottom line
    bottom_row = rows - 2
    while bottom_row > middle_row and grid.get_cell(bottom_row, left_col) == 2:
        bottom_row -= 1
    for c in range(left_col + 1, cols):
        if grid.get_cell(bottom_row, c) == 2:
            break
        grid.set_cell(bottom_row, c, 3)

def ensure_three_edge_contact(grid: ColoredGrid, left_col: int):
    rows, cols = grid.num_rows, grid.num_cols
    edges_touched = sum([
        any(grid.get_cell(0, c) == 3 for c in range(cols)),
        any(grid.get_cell(rows-1, c) == 3 for c in range(cols)),
        any(grid.get_cell(r, 0) == 3 for r in range(rows)),
        any(grid.get_cell(r, cols-1) == 3 for r in range(rows))
    ])
    
    if edges_touched < 3:
        # Extend top line to right edge
        for c in range(cols-1, left_col, -1):
            if grid.get_cell(0, c) == 0:
                grid.set_cell(0, c, 3)
        
        # Extend bottom line to right edge
        for c in range(cols-1, left_col, -1):
            if grid.get_cell(rows-2, c) == 0:
                grid.set_cell(rows-2, c, 3)
        
        # Extend vertical line to top if needed
        if grid.get_cell(0, left_col) == 0:
            grid.set_cell(0, left_col, 3)

def fill_bottom_row(grid: ColoredGrid, left_col: int):
    bottom_row = grid.num_rows - 1
    if all(grid.get_cell(bottom_row, c) != 2 for c in range(grid.num_cols)):
        for c in range(left_col, grid.num_cols):
            if grid.get_cell(bottom_row-1, c) == 3:
                grid.set_cell(bottom_row, c, 3)

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

def optimize_e_shape(grid: ColoredGrid, left_col: int):
    rows, cols = grid.num_rows, grid.num_cols
    
    # Try to extend horizontal lines
    for r in [0, rows // 2, rows - 2]:
        for c in range(cols - 1, left_col, -1):
            if all(grid.get_cell(r, i) == 3 for i in range(left_col, c)) and grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 3)
            else:
                break
    
    # Ensure clear spaces inside the 'E'
    for r in range(1, rows - 1):
        for c in range(left_col + 1, cols - 1):
            if (grid.get_cell(r-1, c) == 3 and grid.get_cell(r+1, c) == 3 and
                grid.get_cell(r, c-1) == 3 and grid.get_cell(r, c+1) == 3):
                grid.set_cell(r, c, 0)
