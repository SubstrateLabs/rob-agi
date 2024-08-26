from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_79369cc6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a new yellow-magenta formation in the least occupied quadrant.
    
    The function divides the grid into four quadrants and analyzes each for the presence of yellow (4) and magenta (6) cells.
    It selects the quadrant with the least yellow and magenta presence, prioritizing quadrants in the order: Q2 > Q3 > Q4 > Q1.
    A new formation (either 2x2 or 3x3) is then created in the bottom-right corner of the chosen quadrant,
    depending on the available space. The formation consists of yellow (4) and magenta (6) cells in a specific pattern.
    Only one such transformation is applied per grid, and the rest of the grid remains unchanged.
    """
    def count_yellow_magenta(quadrant):
        return sum(cell in [4, 6] for row in quadrant for cell in row)

    def get_quadrant(grid, quad):
        rows, cols = grid.get_dimensions()
        mid_row, mid_col = rows // 2, cols // 2
        if quad == 1: return [row[:mid_col] for row in grid[:mid_row]]
        if quad == 2: return [row[mid_col:] for row in grid[:mid_row]]
        if quad == 3: return [row[:mid_col] for row in grid[mid_row:]]
        if quad == 4: return [row[mid_col:] for row in grid[mid_row:]]

    def get_formation_space(quadrant):
        rows, cols = len(quadrant), len(quadrant[0])
        if rows >= 3 and cols >= 3:
            return 3
        elif rows >= 2 and cols >= 2:
            return 2
        return 0

    # Analyze quadrants
    quadrants = [get_quadrant(input_grid, i) for i in range(1, 5)]
    counts = [count_yellow_magenta(q) for q in quadrants]
    spaces = [get_formation_space(q) for q in quadrants]

    # Select target quadrant
    target_quad = min(range(4), key=lambda i: (counts[i], -i))

    # Determine formation size and position
    size = spaces[target_quad]
    rows, cols = input_grid.get_dimensions()
    if target_quad in [0, 1]:
        row = (rows // 2) - size
    else:
        row = rows - size
    if target_quad in [0, 2]:
        col = (cols // 2) - size
    else:
        col = cols - size

    # Create new grid and apply transformation
    new_grid = input_grid.deep_copy()
    new_grid.set_cell(row, col, 4)  # Yellow
    new_grid.set_cell(row + size - 1, col + size - 1, 6)  # Magenta
    if size == 3:
        new_grid.set_cell(row, col + 1, 4)  # Additional Yellow for 3x3
        new_grid.set_cell(row + 1, col, 4)  # Additional Yellow for 3x3

    return new_grid
