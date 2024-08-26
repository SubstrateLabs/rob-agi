from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, Set

def solve_96a8c0cd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a comprehensive tree-like structure
    of red (2) cells connecting all non-black colored cells. The algorithm follows these steps:

    1. Analyze the input grid to find the active area containing colored cells.
    2. Create a new grid, copying all non-black colored cells from the input.
    3. Fill the active area with red cells, preserving original colored cells.
    4. Process the leftmost column and top row specially.
    5. Connect isolated colored cells in the top row and leftmost column.
    6. Extend the red structure to the grid edges.
    7. Perform a final check and cleanup.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the red network connecting colored cells.
    """
    rows, cols = input_grid.get_dimensions()
    grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    def is_colored(r: int, c: int) -> bool:
        return input_grid.get_cell(r, c) > 0

    # Step 1: Analyze the input grid
    colored_cells = [(r, c) for r in range(rows) for c in range(cols) if is_colored(r, c)]
    if not colored_cells:
        return input_grid
    min_row, max_row = min(r for r, _ in colored_cells), max(r for r, _ in colored_cells)
    min_col, max_col = min(c for _, c in colored_cells), max(c for _, c in colored_cells)

    # Step 2: Copy non-black colored cells
    for r, c in colored_cells:
        grid.set_cell(r, c, input_grid.get_cell(r, c))

    # Step 3: Fill active area with red cells
    for r in range(min_row + 1, max_row + 1):
        for c in range(min_col, max_col + 1):
            if not is_colored(r, c):
                grid.set_cell(r, c, 2)

    # Step 4: Process leftmost column and top row
    for r in range(min_row, max_row + 1):
        if is_colored(r, min_col) or (min_col + 1 < cols and grid.get_cell(r, min_col + 1) == 2):
            grid.set_cell(r, min_col, 2)

    for c in range(min_col, max_col + 1):
        if is_colored(min_row, c) or (min_row + 1 < rows and grid.get_cell(min_row + 1, c) == 2):
            grid.set_cell(min_row, c, 2)

    # Step 5: Connect isolated colored cells in top row and leftmost column
    for c in range(min_col, max_col):
        if is_colored(min_row, c) and grid.get_cell(min_row, c + 1) == 0:
            for cc in range(c + 1, max_col + 1):
                if grid.get_cell(min_row, cc) != 0:
                    break
                grid.set_cell(min_row, cc, 2)

    for r in range(min_row, max_row):
        if is_colored(r, min_col) and grid.get_cell(r + 1, min_col) == 0:
            for rr in range(r + 1, max_row + 1):
                if grid.get_cell(rr, min_col) != 0:
                    break
                grid.set_cell(rr, min_col, 2)

    # Step 6: Extend red structure to grid edges
    for c in range(min_col):
        grid.set_cell(max_row, c, 2)
    for r in range(max_row + 1, rows):
        grid.set_cell(r, max_col, 2)

    # Step 7: Final check and cleanup
    for r, c in colored_cells:
        grid.set_cell(r, c, input_grid.get_cell(r, c))

    return grid
