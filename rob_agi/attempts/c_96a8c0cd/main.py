from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, Set

def solve_96a8c0cd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal tree-like structure
    of red (2) cells connecting all non-black colored cells. The algorithm creates
    vertical red lines in columns containing colored cells and connects them with
    horizontal lines where necessary.

    1. Identify columns with colored cells and create vertical red lines.
    2. Connect colored cells to the nearest vertical red line with horizontal lines.
    3. Handle special cases for colored cells in the top row and leftmost column.
    4. Ensure all colored cells are connected to the red network.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the red network connecting colored cells.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_colored(r: int, c: int) -> bool:
        return grid.get_cell(r, c) > 0 and grid.get_cell(r, c) != 2

    # Step 1: Identify columns with colored cells and create vertical red lines
    columns_with_color = set()
    for c in range(cols):
        if any(is_colored(r, c) for r in range(rows)):
            columns_with_color.add(c)
            for r in range(rows):
                if not is_colored(r, c):
                    grid.set_cell(r, c, 2)

    # Add an extra vertical line to the right if needed
    if columns_with_color and max(columns_with_color) < cols - 1:
        rightmost_col = max(columns_with_color) + 1
        for r in range(rows):
            grid.set_cell(r, rightmost_col, 2)
        columns_with_color.add(rightmost_col)

    # Step 2: Connect colored cells to the nearest vertical red line
    for r in range(rows):
        for c in range(cols):
            if is_colored(r, c):
                left = right = c
                while left > 0 and left not in columns_with_color:
                    left -= 1
                while right < cols - 1 and right not in columns_with_color:
                    right += 1
                nearest = left if c - left <= right - c else right
                for cc in range(min(c, nearest), max(c, nearest) + 1):
                    if not is_colored(r, cc):
                        grid.set_cell(r, cc, 2)

    # Step 3: Handle special cases
    # Top row
    for c in range(cols):
        if is_colored(0, c):
            nearest = min(columns_with_color, key=lambda x: abs(x - c))
            for cc in range(min(c, nearest), max(c, nearest) + 1):
                if not is_colored(0, cc):
                    grid.set_cell(0, cc, 2)

    # Leftmost column
    if any(is_colored(r, 0) for r in range(rows)):
        for r in range(rows):
            if not is_colored(r, 0):
                grid.set_cell(r, 0, 2)

    return grid
