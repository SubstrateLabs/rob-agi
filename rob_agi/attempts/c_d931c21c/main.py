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
    Transforms the input grid based on the following rules:
    1. Blue (1) cells remain unchanged.
    2. Black (0) cells:
       - If they have at least one blue neighbor, they become red (2).
       - If they have at least one red neighbor (after applying the previous rule), they become green (3).
       - Otherwise, they remain black.
    3. All other colored cells remain unchanged.
    4. If no changes occur after applying these rules, the original grid is returned.

    The transformation is applied only once, not iteratively.
    Neighbors are considered in all 8 directions (including diagonals).
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    changes_made = False

    def get_neighbors(row: int, col: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    neighbors.append((new_row, new_col))
        return neighbors

    # First pass: change black cells to red if they have blue neighbors
    for row in range(rows):
        for col in range(cols):
            if new_grid.get_cell(row, col) == 0:  # Black cell
                neighbors = get_neighbors(row, col)
                if any(new_grid.get_cell(r, c) == 1 for r, c in neighbors):
                    new_grid.set_cell(row, col, 2)  # Change to red
                    changes_made = True

    # Second pass: change black cells to green if they have red neighbors
    for row in range(rows):
        for col in range(cols):
            if new_grid.get_cell(row, col) == 0:  # Black cell
                neighbors = get_neighbors(row, col)
                if any(new_grid.get_cell(r, c) == 2 for r, c in neighbors):
                    new_grid.set_cell(row, col, 3)  # Change to green
                    changes_made = True

    return new_grid if changes_made else input_grid
