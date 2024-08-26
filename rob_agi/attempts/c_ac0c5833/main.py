from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac0c5833(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding red (2) regions into 3x3 areas with one corner missing.
    The expansion respects yellow (4) cells as barriers and maintains the pattern of leaving
    one corner empty when possible. Overlapping expansions are merged consistently.
    Yellow cells act as anchors for new red expansions, and existing red structures are preserved.
    """
    grid = input_grid.deep_copy()
    expand_red_regions(grid, input_grid)
    return grid

def expand_red_regions(grid: ColoredGrid, original_grid: ColoredGrid):
    for row in range(grid.num_rows):
        for col in range(grid.num_cols):
            if original_grid.values[row][col] == 4:  # Yellow cell
                expand_around_yellow(grid, original_grid, row, col)
            elif original_grid.values[row][col] == 2:  # Red cell
                expand_existing_red(grid, original_grid, row, col)

def expand_around_yellow(grid: ColoredGrid, original_grid: ColoredGrid, row: int, col: int):
    area = get_3x3_area(grid, row, col)
    corner_to_remove = choose_corner_to_remove(original_grid, area)
    
    for r, c in area:
        if original_grid.values[r][c] == 0 and (r, c) != corner_to_remove:
            grid.values[r][c] = 2

def expand_existing_red(grid: ColoredGrid, original_grid: ColoredGrid, row: int, col: int):
    area = get_3x3_area(grid, row, col)
    corner_to_remove = choose_corner_to_remove(original_grid, area)
    
    for r, c in area:
        if grid.values[r][c] == 0 and original_grid.values[r][c] != 4 and (r, c) != corner_to_remove:
            grid.values[r][c] = 2

def get_3x3_area(grid: ColoredGrid, row: int, col: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(max(0, row-1), min(grid.num_rows, row+2))
            for c in range(max(0, col-1), min(grid.num_cols, col+2))]

def choose_corner_to_remove(original_grid: ColoredGrid, area: List[Tuple[int, int]]) -> Tuple[int, int]:
    corners = [area[0], area[-1], area[0], area[-1]]  # Adjust based on area size
    for corner in corners:
        if original_grid.values[corner[0]][corner[1]] == 0:
            return corner
    return area[0]  # Default to first cell if no empty corner
