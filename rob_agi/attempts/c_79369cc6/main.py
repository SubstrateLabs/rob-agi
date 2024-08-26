from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79369cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating or expanding a yellow-magenta formation.
    
    The function scans the entire grid and evaluates potential positions for a
    yellow-magenta formation. It selects the best position based on a scoring system
    that considers existing magenta and yellow squares, as well as the potential for
    expansion. The chosen area is then transformed by adding yellow squares (4) above
    and to the left of a magenta square (6), creating or expanding a 2x2 or 3x3
    yellow-magenta formation. Only one such transformation is applied per grid,
    and the rest of the grid remains unchanged.
    """
    def evaluate_formation(grid: ColoredGrid, row: int, col: int) -> Tuple[int, int]:
        score = 0
        size = 2  # Start with assuming a 2x2 formation
        for dr in range(-2, 1):
            for dc in range(-2, 1):
                r, c = row + dr, col + dc
                if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                    if grid[r][c] == 6:  # Magenta
                        score += 3
                    elif grid[r][c] == 4:  # Yellow
                        score += 2
                    elif grid[r][c] in [0, 8]:  # Easily replaceable colors
                        score += 1
                        if dr < 0 and dc < 0:
                            size = 3  # Can form a 3x3 formation
        return score, size

    best_score = -1
    best_pos = None
    best_size = 0
    
    for row in range(input_grid.num_rows):
        for col in range(input_grid.num_cols):
            score, size = evaluate_formation(input_grid, row, col)
            if score > best_score:
                best_score = score
                best_pos = (row, col)
                best_size = size
    
    output_grid = input_grid.deep_copy()
    
    if best_pos:
        row, col = best_pos
        output_grid.set_cell(row, col, 6)  # Ensure anchor is magenta
        output_grid.set_cell(row-1, col, 4)  # Yellow above
        output_grid.set_cell(row, col-1, 4)  # Yellow to the left
        if best_size == 3:
            output_grid.set_cell(row-1, col-1, 4)  # Yellow diagonally up-left
    
    return output_grid
