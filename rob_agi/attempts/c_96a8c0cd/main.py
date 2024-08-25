from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, Set

def solve_96a8c0cd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal tree-like structure
    of red (2) cells connecting all non-black colored cells. The algorithm follows these steps:

    1. Analyze the input grid to find colored cells and their positions.
    2. Create vertical red lines in columns with colored cells.
    3. Add a base horizontal red line below the bottommost colored cell.
    4. Connect isolated colored cells to the nearest vertical red line.
    5. Handle special cases for colored cells in the top row and leftmost column.
    6. Optimize the structure by removing redundant red cells.
    7. Ensure all colored cells are connected to the red network.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the red network connecting colored cells.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_colored(r: int, c: int) -> bool:
        return grid.get_cell(r, c) > 0 and grid.get_cell(r, c) != 2

    # Step 1: Analyze the input grid
    colored_cells = [(r, c) for r in range(rows) for c in range(cols) if is_colored(r, c)]
    if not colored_cells:
        return grid
    min_row, max_row = min(r for r, _ in colored_cells), max(r for r, _ in colored_cells)
    min_col, max_col = min(c for _, c in colored_cells), max(c for _, c in colored_cells)

    # Step 2: Create vertical red lines
    columns_with_color = set(c for _, c in colored_cells)
    for c in columns_with_color:
        for r in range(min_row, max_row + 1):
            if not is_colored(r, c):
                grid.set_cell(r, c, 2)

    # Step 3: Add base horizontal red line
    base_row = min(max_row + 1, rows - 1)
    for c in range(min_col, max_col + 1):
        if not is_colored(base_row, c):
            grid.set_cell(base_row, c, 2)

    # Step 4: Connect isolated colored cells
    for r, c in colored_cells:
        if c not in columns_with_color:
            left = right = c
            while left > min_col and left not in columns_with_color:
                left -= 1
            while right < max_col and right not in columns_with_color:
                right += 1
            nearest = left if c - left <= right - c else right
            for cc in range(min(c, nearest), max(c, nearest) + 1):
                if not is_colored(r, cc):
                    grid.set_cell(r, cc, 2)

    # Step 5: Handle special cases
    # Top row
    for c in range(min_col, max_col + 1):
        if is_colored(min_row, c) and c not in columns_with_color:
            nearest = min(columns_with_color, key=lambda x: abs(x - c))
            for cc in range(min(c, nearest), max(c, nearest) + 1):
                if not is_colored(min_row, cc):
                    grid.set_cell(min_row, cc, 2)

    # Leftmost column
    if min_col == 0 and 0 not in columns_with_color:
        for r in range(min_row, base_row + 1):
            if not is_colored(r, 0):
                grid.set_cell(r, 0, 2)

    # Step 6: Optimize the structure (remove redundant red cells)
    for r in range(rows):
        last_colored = -1
        for c in range(cols):
            if is_colored(r, c) or (grid.get_cell(r, c) == 2 and (r == base_row or c in columns_with_color)):
                last_colored = c
            elif grid.get_cell(r, c) == 2 and c > last_colored:
                grid.set_cell(r, c, 0)

    return grid
