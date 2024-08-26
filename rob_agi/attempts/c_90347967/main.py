from rob_agi.colored_grid import ColoredGrid

def solve_90347967(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rotating it 90 degrees clockwise and moving non-black cells to the top-right corner.
    
    1. Identifies all non-black cells in the input grid.
    2. Rotates these cells 90 degrees clockwise.
    3. Sorts the rotated cells based on new column (descending) and new row (ascending).
    4. Places the sorted cells into a new grid, starting from the top-right corner and moving left.
    5. Within each new column, cells are placed from top to bottom.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Identify and rotate non-black cells
    rotated_cells = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                new_r, new_c = c, (rows - 1) - r
                rotated_cells.append((new_r, new_c, input_grid.values[r][c]))

    # Sort rotated cells
    rotated_cells.sort(key=lambda x: (-x[1], x[0]))

    # Calculate starting column for placement
    unique_cols = len(set(cell[1] for cell in rotated_cells))
    start_col = cols - unique_cols

    # Place cells in new grid
    current_col = start_col
    prev_new_col = None
    for new_r, new_c, color in rotated_cells:
        if new_c != prev_new_col:
            current_col += 1
            current_row = 0
        new_grid.values[current_row][current_col] = color
        current_row += 1
        prev_new_col = new_c

    return new_grid
