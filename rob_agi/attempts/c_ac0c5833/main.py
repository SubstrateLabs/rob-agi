from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac0c5833(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding red (2) regions into 3x3 areas around yellow (4) cells,
    with one corner missing. The expansion preserves existing structures, consistently chooses
    which corner to leave empty, and ensures connectivity of red regions. Yellow cells act as
    anchors for new red expansions, and existing red and yellow structures are preserved.
    The function also handles overlapping expansions and prevents isolated red cells.
    """
    grid = input_grid.deep_copy()
    yellow_cells = [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 4]
    
    for row, col in yellow_cells:
        expand_around_yellow(grid, row, col)
    
    connect_expansions(grid)
    remove_isolated_reds(grid)
    
    return grid

def expand_around_yellow(grid: ColoredGrid, row: int, col: int):
    corners = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
    empty_corner = choose_empty_corner(grid, corners)
    
    for i in range(max(0, row-1), min(grid.num_rows, row+2)):
        for j in range(max(0, col-1), min(grid.num_cols, col+2)):
            if (i, j) != (row, col) and (i, j) != empty_corner:
                if grid.values[i][j] == 0:  # Only change if it's currently black
                    grid.values[i][j] = 2  # Change to red

def choose_empty_corner(grid: ColoredGrid, corners: List[Tuple[int, int]]) -> Tuple[int, int]:
    valid_corners = [c for c in corners if 0 <= c[0] < grid.num_rows and 0 <= c[1] < grid.num_cols]
    empty_corners = [c for c in valid_corners if grid.values[c[0]][c[1]] == 0]
    return empty_corners[0] if empty_corners else valid_corners[0]

def connect_expansions(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2:
                connect_neighbors(grid, r, c)

def connect_neighbors(grid: ColoredGrid, row: int, col: int):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
            if grid.values[nr][nc] == 0:
                diagonal_reds = sum(1 for ddr, ddc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                                    if 0 <= nr+ddr < grid.num_rows and 0 <= nc+ddc < grid.num_cols
                                    and grid.values[nr+ddr][nc+ddc] == 2)
                if diagonal_reds >= 2:
                    grid.values[nr][nc] = 2

def remove_isolated_reds(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2 and is_isolated(grid, r, c):
                grid.values[r][c] = 0

def is_isolated(grid: ColoredGrid, row: int, col: int) -> bool:
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
            if grid.values[nr][nc] in [2, 4]:
                return False
    return True
