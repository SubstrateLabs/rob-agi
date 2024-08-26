from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_ac0c5833(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding red (2) regions and creating new red regions around yellow (4) cells
    adjacent to red. The process continues until no further expansion is possible. Yellow cells act as barriers
    and remain unchanged. The expansion occurs in all eight directions (including diagonals).
    """
    grid = input_grid.deep_copy()
    red_cells, yellow_cells = identify_colored_cells(grid)
    
    # Initial expansion
    for red_cell in red_cells:
        flood_fill(grid, red_cell)
    
    # Expand around yellow cells
    while True:
        new_red_cells = identify_yellow_adjacent_to_red(grid, yellow_cells)
        if not new_red_cells:
            break
        for cell in new_red_cells:
            expand_around_yellow(grid, cell)
    
    return grid

def identify_colored_cells(grid: ColoredGrid) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    red_cells = []
    yellow_cells = []
    for i in range(grid.num_rows):
        for j in range(grid.num_cols):
            if grid.values[i][j] == 2:
                red_cells.append((i, j))
            elif grid.values[i][j] == 4:
                yellow_cells.append((i, j))
    return red_cells, yellow_cells

def flood_fill(grid: ColoredGrid, start: Tuple[int, int]):
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.num_rows and 0 <= ny < grid.num_cols and grid.values[nx][ny] == 0:
                grid.values[nx][ny] = 2
                stack.append((nx, ny))

def identify_yellow_adjacent_to_red(grid: ColoredGrid, yellow_cells: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    new_red_cells = set()
    for x, y in yellow_cells:
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.num_rows and 0 <= ny < grid.num_cols and grid.values[nx][ny] == 2:
                new_red_cells.add((x, y))
                break
    return new_red_cells

def expand_around_yellow(grid: ColoredGrid, yellow_cell: Tuple[int, int]):
    x, y = yellow_cell
    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.num_rows and 0 <= ny < grid.num_cols and grid.values[nx][ny] == 0:
            flood_fill(grid, (nx, ny))
