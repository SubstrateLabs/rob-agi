from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac0c5833(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding red (2) regions into 3x3 areas with one corner missing.
    The expansion respects yellow (4) cells as barriers and maintains the pattern of leaving
    one corner empty when possible. Overlapping expansions are merged consistently.
    """
    grid = input_grid.deep_copy()
    original_red_cells = find_red_cells(input_grid)
    
    # First pass: expand each original red cell
    for cell in original_red_cells:
        expand_3x3(grid, input_grid, cell)
    
    # Second pass: merge overlapping areas
    for row in range(grid.num_rows):
        for col in range(grid.num_cols):
            if grid.values[row][col] == 2:
                merge_3x3(grid, input_grid, (row, col))
    
    return grid

def find_red_cells(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(row, col) for row in range(grid.num_rows) for col in range(grid.num_cols) if grid.values[row][col] == 2]

def get_3x3_area(grid: ColoredGrid, row: int, col: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(max(0, row-1), min(grid.num_rows, row+2))
            for c in range(max(0, col-1), min(grid.num_cols, col+2))]

def count_original_red(original_grid: ColoredGrid, area: List[Tuple[int, int]]) -> int:
    return sum(1 for r, c in area if original_grid.values[r][c] == 2)

def choose_corner_to_remove(original_grid: ColoredGrid, area: List[Tuple[int, int]]) -> Tuple[int, int]:
    corners = [area[0], area[2], area[6], area[8]]
    for corner in corners:
        if original_grid.values[corner[0]][corner[1]] != 2:
            return corner
    return corners[0]  # Default to top-left if all corners were originally red

def expand_3x3(grid: ColoredGrid, original_grid: ColoredGrid, cell: Tuple[int, int]):
    area = get_3x3_area(grid, *cell)
    red_count = count_original_red(original_grid, area)
    corner_to_remove = choose_corner_to_remove(original_grid, area) if red_count < 8 else None
    
    for r, c in area:
        if original_grid.values[r][c] != 4 and (r, c) != corner_to_remove:
            grid.values[r][c] = 2

def merge_3x3(grid: ColoredGrid, original_grid: ColoredGrid, cell: Tuple[int, int]):
    area = get_3x3_area(grid, *cell)
    corner_to_remove = choose_corner_to_remove(original_grid, area)
    
    for r, c in area:
        if grid.values[r][c] == 0 and original_grid.values[r][c] != 4 and (r, c) != corner_to_remove:
            grid.values[r][c] = 2
