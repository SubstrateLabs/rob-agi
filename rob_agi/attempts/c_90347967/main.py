from rob_agi.colored_grid import ColoredGrid

def solve_90347967(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving non-black cells to the right side of the grid.
    The transformation maintains the relative vertical order of cells from the same original column
    and arranges them from right to left based on their original column order.
    
    1. Identifies all non-black cells in the input grid.
    2. Sorts these cells based on their original column index.
    3. Places the sorted cells into a new grid, starting from the rightmost column and moving left.
    4. Within each new column, cells are placed from bottom to top.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Identify non-black cells
    non_black_cells = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                non_black_cells.append((r, c, input_grid.values[r][c]))

    # Sort non-black cells based on original column index
    non_black_cells.sort(key=lambda x: x[1])

    # Calculate starting column for placement
    unique_cols = len(set(cell[1] for cell in non_black_cells))
    start_col = cols - unique_cols

    # Place cells in new grid
    current_col = start_col
    prev_orig_col = -1
    for _, c, color in non_black_cells:
        if c != prev_orig_col:
            current_col += 1
            current_row = rows - 1
        new_grid.values[current_row][current_col] = color
        current_row -= 1
        prev_orig_col = c

    return new_grid
