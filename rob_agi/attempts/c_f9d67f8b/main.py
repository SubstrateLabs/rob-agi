from rob_agi.colored_grid import ColoredGrid
from collections import deque
import random

def solve_f9d67f8b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Identifying and removing all brown (9) cells.
    2. Filling the empty spaces with colors from neighboring cells.
    3. Ensuring all spaces are filled with valid colors.

    Args:
        input_grid (ColoredGrid): The input grid to transform.

    Returns:
        ColoredGrid: The transformed grid with brown cells removed and spaces filled.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()
    empty_cells = []

    # Step 1: Identify and remove brown cells
    for r in range(rows):
        for c in range(cols):
            if new_grid.get_cell(r, c) == 9:  # Brown color
                new_grid.set_cell(r, c, -1)  # Mark as empty
                empty_cells.append((r, c))

    # Step 2: Fill empty spaces
    while empty_cells:
        r, c = empty_cells.pop(0)
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and new_grid.get_cell(nr, nc) != -1:
                neighbors.append(new_grid.get_cell(nr, nc))
        if neighbors:
            new_grid.set_cell(r, c, random.choice(neighbors))
        else:
            empty_cells.append((r, c))  # Re-add to queue if no valid neighbors

    # Step 3: Final check
    for r in range(rows):
        for c in range(cols):
            if new_grid.get_cell(r, c) == -1:
                # Find nearest non-empty neighbor
                for d in range(1, max(rows, cols)):
                    for dr, dc in [(0, d), (d, 0), (0, -d), (-d, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and new_grid.get_cell(nr, nc) != -1:
                            new_grid.set_cell(r, c, new_grid.get_cell(nr, nc))
                            break
                    if new_grid.get_cell(r, c) != -1:
                        break

    return new_grid
