from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_13713586(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions downwards and to the right,
    while preserving gray boundaries and respecting the "first to reach" rule.
    
    The algorithm works as follows:
    1. Create a copy of the input grid.
    2. Identify all colored positions (excluding black and gray).
    3. Sort colored positions from top to bottom, then left to right.
    4. For each colored position, expand downwards and then to the right.
    5. Preserve gray boundaries by restoring them after expansion.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def expand_color(r: int, c: int, color: int):
        # Expand downwards
        last_row = r
        while last_row + 1 < rows and grid.values[last_row + 1][c] == 0:
            last_row += 1
            grid.values[last_row][c] = color

        # Expand rightwards for each row
        for row in range(r, last_row + 1):
            col = c
            while col + 1 < cols and (grid.values[row][col + 1] == 0 or grid.values[row][col + 1] == color):
                col += 1
                grid.values[row][col] = color

    # Identify colored positions
    colored_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] not in [0, 5]:
                colored_positions.append((r, c))

    # Sort colored positions
    colored_positions.sort()

    # Expand colors
    for r, c in colored_positions:
        color = grid.values[r][c]
        expand_color(r, c, color)

    # Preserve gray boundaries
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 5:
                grid.values[r][c] = 5

    return grid
