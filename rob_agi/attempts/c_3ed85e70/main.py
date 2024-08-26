from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

color_sequence = {1: 2, 2: 4, 4: 8, 8: 1}

def find_expandable_cells(grid: ColoredGrid, unchangeable: Set[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    expandable = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in unchangeable and grid.values[r][c] in color_sequence:
                expandable.append((r, c, grid.values[r][c]))
    return expandable

def expand_cell(grid: ColoredGrid, row: int, col: int, color: int, unchangeable: Set[Tuple[int, int]]) -> None:
    next_color = color_sequence[color]
    rows, cols = grid.get_dimensions()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols and (r, c) not in unchangeable:
                if grid.values[r][c] == 0:
                    grid.values[r][c] = color
    grid.values[row][col] = next_color

def solve_3ed85e70(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by expanding color patterns.
    
    The solution works as follows:
    1. Identify cells with colors in the sequence (1, 2, 4, 8) that are not in the unchangeable area.
    2. For each identified cell:
       a. Change surrounding black (0) cells to the current color.
       b. Change the center cell to the next color in the sequence.
    3. Repeat until no more changes can be made.
    
    Color sequence: 1 (blue) -> 2 (red) -> 4 (yellow) -> 8 (sky) -> 1 (blue)
    """
    grid = input_grid.deep_copy()
    unchangeable = set((r, c) for r, row in enumerate(grid.values) 
                       for c, val in enumerate(row) if val == 3)
    
    while True:
        original = grid.deep_copy()
        expandable_cells = find_expandable_cells(grid, unchangeable)
        
        for row, col, color in expandable_cells:
            expand_cell(grid, row, col, color, unchangeable)
        
        if grid.values == original.values:
            break
    
    return grid
