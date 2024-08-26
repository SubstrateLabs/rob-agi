from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79369cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding or creating a yellow-magenta formation.
    
    The function scans the grid for magenta squares (6) and evaluates 3x3 areas
    with the magenta square at the bottom-right corner. It selects the best area
    based on a scoring system that considers existing magenta and yellow squares,
    as well as potential for expansion. The chosen area is then transformed by
    adding yellow squares (4) above and to the left of magenta squares, creating
    or expanding a yellow-magenta formation. Only one such transformation is
    applied per grid, and the rest of the grid remains unchanged.
    """
    def evaluate_area(grid: ColoredGrid, row: int, col: int) -> float:
        score = 0
        for dr in range(-2, 1):
            for dc in range(-2, 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    if grid[nr][nc] == 6:
                        score += 2
                    elif grid[nr][nc] == 4:
                        score += 1
                    elif grid[nr][nc] == 0 and (dr < 0 or dc < 0):
                        score += 1
        return score

    def transform_area(grid: ColoredGrid, row: int, col: int):
        for dr in range(-2, 1):
            for dc in range(-2, 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    if grid[nr][nc] == 6:
                        continue
                    elif grid[nr][nc] == 0 and (dr < 0 or dc < 0):
                        grid.set_cell(nr, nc, 4)

    output_grid = input_grid.deep_copy()
    best_score = -1
    best_area = None

    for row in range(input_grid.num_rows):
        for col in range(input_grid.num_cols):
            if input_grid[row][col] == 6:
                score = evaluate_area(input_grid, row, col)
                if score > best_score:
                    best_score = score
                    best_area = (row, col)

    if best_area:
        transform_area(output_grid, best_area[0], best_area[1])

    return output_grid
