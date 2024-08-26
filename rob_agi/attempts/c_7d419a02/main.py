from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7d419a02(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing some blue (8) regions to yellow (4).
    
    The transformation follows these rules:
    1. Blue regions of 2x2 or larger that touch black (0) cells are changed to yellow.
    2. Single-width blue lines and regions not touching black remain blue.
    3. Black (0) and magenta (6) cells remain unchanged.
    4. The transformation is applied consistently across the entire grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    for row in range(rows):
        for col in range(cols):
            if is_blue(grid.values[row][col]):
                if is_2x2_or_larger_blue(grid, row, col):
                    if touches_black(grid, row, col) and not is_part_of_larger_blue(grid, row, col):
                        change_region_to_yellow(grid, row, col)

    return grid

def is_blue(cell: int) -> bool:
    return cell == 8

def is_black(cell: int) -> bool:
    return cell == 0

def touches_black(grid: ColoredGrid, row: int, col: int) -> bool:
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < grid.num_rows and 0 <= new_col < grid.num_cols:
                if is_black(grid.values[new_row][new_col]):
                    return True
    return False

def is_2x2_or_larger_blue(grid: ColoredGrid, row: int, col: int) -> bool:
    if row + 1 < grid.num_rows and col + 1 < grid.num_cols:
        return (is_blue(grid.values[row][col]) and
                is_blue(grid.values[row][col+1]) and
                is_blue(grid.values[row+1][col]) and
                is_blue(grid.values[row+1][col+1]))
    return False

def is_part_of_larger_blue(grid: ColoredGrid, row: int, col: int) -> bool:
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < grid.num_rows and 0 <= new_col < grid.num_cols:
                if is_blue(grid.values[new_row][new_col]) and not touches_black(grid, new_row, new_col):
                    return True
    return False

def change_region_to_yellow(grid: ColoredGrid, start_row: int, start_col: int):
    stack = [(start_row, start_col)]
    while stack:
        row, col = stack.pop()
        if is_blue(grid.values[row][col]):
            grid.values[row][col] = 4  # Change to yellow
            # Add neighboring blue cells to stack
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < grid.num_rows and 0 <= new_col < grid.num_cols:
                    if is_blue(grid.values[new_row][new_col]):
                        stack.append((new_row, new_col))
