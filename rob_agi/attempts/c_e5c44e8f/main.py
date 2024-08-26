from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e5c44e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by adding a green 'E' pattern based on the following rules:
    1. Locate the initial green (3) square in the input grid.
    2. Create an 'E' pattern template, starting with 9x7 size.
    3. Find the optimal position for the template, ensuring it includes the initial green square
       and doesn't overwrite any red squares.
    4. If no valid position is found, reduce the template size from the bottom up.
    5. Place the template on the grid, filling with green (3) squares.
    6. Preserve any red squares and areas outside the template.

    The function adapts to various edge cases and aims to create the largest possible 'E'
    pattern that fits within the constraints of the input grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    # Find the initial green square
    initial_green = None
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 3:
                initial_green = (r, c)
                break
        if initial_green:
            break

    if not initial_green:
        return output_grid

    # Create the 'E' pattern template
    template = create_e_template()

    # Find the optimal position for the template
    position = find_optimal_position(input_grid, template, initial_green)

    # If no valid position is found, return the input grid
    if not position:
        return output_grid

    # Place the template on the grid
    place_template(output_grid, template, position)

    return output_grid

def create_e_template() -> List[List[int]]:
    return [
        [0, 3, 0, 3, 3, 3, 3],
        [0, 3, 0, 3, 0, 0, 0],
        [0, 3, 0, 3, 0, 0, 0],
        [0, 3, 0, 3, 3, 3, 0],
        [0, 3, 0, 3, 0, 0, 0],
        [0, 3, 0, 3, 0, 0, 0],
        [0, 3, 0, 3, 3, 3, 3],
        [0, 3, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 3, 3, 3]
    ]

def find_optimal_position(grid: ColoredGrid, template: List[List[int]], initial_green: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = grid.get_dimensions()
    template_rows, template_cols = len(template), len(template[0])

    for r in range(rows - template_rows + 1):
        for c in range(cols - template_cols + 1):
            if is_valid_position(grid, template, r, c, initial_green):
                return r, c

    # If no valid position found, try reducing the template size
    while template_rows > 3:
        template.pop()
        template_rows -= 1
        for r in range(rows - template_rows + 1):
            for c in range(cols - template_cols + 1):
                if is_valid_position(grid, template, r, c, initial_green):
                    return r, c

    return None

def is_valid_position(grid: ColoredGrid, template: List[List[int]], row: int, col: int, initial_green: Tuple[int, int]) -> bool:
    rows, cols = grid.get_dimensions()
    template_rows, template_cols = len(template), len(template[0])

    if not (row <= initial_green[0] < row + template_rows and col <= initial_green[1] < col + template_cols):
        return False

    for r in range(template_rows):
        for c in range(template_cols):
            if template[r][c] == 3:
                grid_r, grid_c = row + r, col + c
                if grid_r >= rows or grid_c >= cols:
                    return False
                if grid.get_cell(grid_r, grid_c) == 2:  # Red square
                    return False

    return True

def place_template(grid: ColoredGrid, template: List[List[int]], position: Tuple[int, int]):
    row, col = position
    for r in range(len(template)):
        for c in range(len(template[0])):
            if template[r][c] == 3 and grid.get_cell(row + r, col + c) != 2:
                grid.set_cell(row + r, col + c, 3)
