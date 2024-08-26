from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def apply_rule(grid: ColoredGrid) -> ColoredGrid:
    new_grid = grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:  # Blue stays blue
                continue
            elif grid.get_cell(r, c) == 0:  # Black
                neighbors = get_neighbors(grid, r, c)
                if any(neighbor == 1 for neighbor in neighbors):
                    new_grid.set_cell(r, c, 2)  # Set to red
                elif any(neighbor == 2 for neighbor in neighbors):
                    new_grid.set_cell(r, c, 3)  # Set to green
    
    return new_grid

def get_neighbors(grid: ColoredGrid, r: int, c: int) -> List[int]:
    rows, cols = grid.get_dimensions()
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append(grid.get_cell(nr, nc))
    return neighbors

def are_grids_identical(grid1: ColoredGrid, grid2: ColoredGrid) -> bool:
    if grid1.get_dimensions() != grid2.get_dimensions():
        return False
    rows, cols = grid1.get_dimensions()
    return all(grid1.get_cell(r, c) == grid2.get_cell(r, c) for r in range(rows) for c in range(cols))

def solve_d931c21c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a cellular automaton-like rule iteratively until the grid stabilizes.
    
    The rule is as follows:
    1. Blue (1) cells remain blue.
    2. Black (0) cells:
       - If they have at least one blue neighbor, they become red (2).
       - If they have at least one red neighbor, they become green (3).
       - Otherwise, they remain black.
    3. The process repeats until the grid no longer changes.
    
    Neighbors are considered in all 8 directions (including diagonals).
    """
    prev_grid = input_grid
    max_iterations = 100  # Safety mechanism to prevent infinite loops
    
    for _ in range(max_iterations):
        new_grid = apply_rule(prev_grid)
        if are_grids_identical(new_grid, prev_grid):
            return new_grid
        prev_grid = new_grid
    
    return prev_grid  # Return the last state if max iterations reached
