from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79369cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding a new yellow-magenta formation in the quadrant that results in the most balanced distribution of yellow (4) and magenta (6) cells.

    The function divides the grid into four quadrants and analyzes each for the presence of yellow and magenta cells.
    It simulates adding a new formation (either 2x2 or 3x3) to each quadrant and chooses the one that results in the most balanced distribution across the entire grid.
    The new formation is placed in the corner of the chosen quadrant closest to the grid center.
    The formation consists of yellow (4) and magenta (6) cells in a specific pattern, avoiding overwriting existing yellow or magenta cells.
    Only one such transformation is applied per grid, and the rest of the grid remains unchanged.
    """
    def count_yellow_magenta(grid):
        return sum(cell in [4, 6] for row in grid.values for cell in row)

    def get_formation_space(grid, row, col):
        rows, cols = grid.get_dimensions()
        if row + 2 < rows and col + 2 < cols:
            return 3
        elif row + 1 < rows and col + 1 < cols:
            return 2
        return 0

    def simulate_addition(grid, row, col, size):
        new_grid = grid.deep_copy()
        add_formation(new_grid, row, col, size)
        return count_yellow_magenta(new_grid)

    def add_formation(grid, row, col, size):
        if size == 3:
            cells = [(row, col), (row, col+1), (row+1, col), (row+1, col+1), (row+1, col+2), (row+2, col+1), (row+2, col+2)]
            for r, c in cells:
                if grid.get_cell(r, c) not in [4, 6]:
                    grid.set_cell(r, c, 4 if (r, c) in cells[:3] else 6)
        elif size == 2:
            cells = [(row, col), (row, col+1), (row+1, col), (row+1, col+1)]
            for r, c in cells:
                if grid.get_cell(r, c) not in [4, 6]:
                    grid.set_cell(r, c, 4 if (r, c) in [(row, col), (row+1, col+1)] else 6)

    rows, cols = input_grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2
    quadrants = [(0, 0), (0, mid_col), (mid_row, 0), (mid_row, mid_col)]

    initial_count = count_yellow_magenta(input_grid)
    best_score = float('inf')
    best_quadrant = None
    best_size = None

    for i, (row, col) in enumerate(quadrants):
        size = get_formation_space(input_grid, row, col)
        if size > 0:
            new_count = simulate_addition(input_grid, row, col, size)
            score = abs(new_count - initial_count)
            if score < best_score or (score == best_score and size > best_size):
                best_score = score
                best_quadrant = i
                best_size = size

    if best_quadrant is not None:
        new_grid = input_grid.deep_copy()
        row, col = quadrants[best_quadrant]
        add_formation(new_grid, row, col, best_size)
        return new_grid
    else:
        return input_grid
