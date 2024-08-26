from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_e5c44e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by adding a green 'E' pattern based on the following rules:
    1. Locate the initial green (3) square in the input grid.
    2. Create an 'E' pattern template, starting with the largest possible size.
    3. Find the optimal position for the template, ensuring it includes the initial green square
       and doesn't overwrite any red squares.
    4. If no valid position is found, reduce the template size from the bottom up.
    5. Place the template on the grid, filling with green (3) squares.
    6. Extend the 'E' pattern to the grid edges if possible.
    7. Fill the bottom row with green squares if possible.
    8. Preserve any red squares and ensure all green squares are connected.

    The function adapts to various edge cases and aims to create the largest possible 'E'
    pattern that fits within the constraints of the input grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    initial_green = find_initial_green(output_grid)
    if not initial_green:
        return output_grid

    template = create_largest_e_template(rows, cols)
    position = find_optimal_position(output_grid, template, initial_green)

    while not position and len(template) > 3:
        template = reduce_template(template)
        position = find_optimal_position(output_grid, template, initial_green)

    if not position:
        return output_grid

    place_template(output_grid, template, position)
    extend_e_pattern(output_grid, position, len(template), len(template[0]))
    fill_bottom_row(output_grid)

    return output_grid

def find_initial_green(grid: ColoredGrid) -> Optional[Tuple[int, int]]:
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                return r, c
    return None

def create_largest_e_template(rows: int, cols: int) -> List[List[int]]:
    max_height = min(rows, 9)
    max_width = min(cols, 7)
    template = [[0] * max_width for _ in range(max_height)]
    
    # Vertical line
    for r in range(max_height):
        template[r][1] = 3
    
    # Horizontal lines
    for c in range(1, max_width):
        template[0][c] = 3
        if max_height > 3:
            template[max_height // 2][c] = 3
        template[-1][c] = 3
    
    return template

def find_optimal_position(grid: ColoredGrid, template: List[List[int]], initial_green: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    rows, cols = grid.num_rows, grid.num_cols
    template_rows, template_cols = len(template), len(template[0])

    for r in range(rows - template_rows + 1):
        for c in range(cols - template_cols + 1):
            if is_valid_position(grid, template, r, c, initial_green):
                return r, c
    return None

def is_valid_position(grid: ColoredGrid, template: List[List[int]], row: int, col: int, initial_green: Tuple[int, int]) -> bool:
    if not (row <= initial_green[0] < row + len(template) and col <= initial_green[1] < col + len(template[0])):
        return False

    for r in range(len(template)):
        for c in range(len(template[0])):
            if template[r][c] == 3:
                grid_r, grid_c = row + r, col + c
                if grid_r >= grid.num_rows or grid_c >= grid.num_cols or grid.get_cell(grid_r, grid_c) == 2:
                    return False
    return True

def reduce_template(template: List[List[int]]) -> List[List[int]]:
    return template[:-1]

def place_template(grid: ColoredGrid, template: List[List[int]], position: Tuple[int, int]):
    row, col = position
    for r in range(len(template)):
        for c in range(len(template[0])):
            if template[r][c] == 3 and grid.get_cell(row + r, col + c) != 2:
                grid.set_cell(row + r, col + c, 3)

def extend_e_pattern(grid: ColoredGrid, position: Tuple[int, int], height: int, width: int):
    row, col = position
    
    # Extend top horizontal line
    for c in range(col + width, grid.num_cols):
        if grid.get_cell(row, c) == 0:
            grid.set_cell(row, c, 3)
        else:
            break
    
    # Extend bottom horizontal line
    for c in range(col + width, grid.num_cols):
        if grid.get_cell(row + height - 1, c) == 0:
            grid.set_cell(row + height - 1, c, 3)
        else:
            break
    
    # Extend vertical line
    for r in range(row + height, grid.num_rows):
        if grid.get_cell(r, col + 1) == 0:
            grid.set_cell(r, col + 1, 3)
        else:
            break

def fill_bottom_row(grid: ColoredGrid):
    bottom_row = grid.num_rows - 1
    for c in range(grid.num_cols):
        if grid.get_cell(bottom_row, c) == 0 and is_connected_to_green(grid, bottom_row, c):
            grid.set_cell(bottom_row, c, 3)

def is_connected_to_green(grid: ColoredGrid, row: int, col: int) -> bool:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.get_cell(nr, nc) == 3:
            return True
    return False
