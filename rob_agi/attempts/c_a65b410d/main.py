from rob_agi.colored_grid import ColoredGrid

def solve_a65b410d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a triangular pattern based on a row of red (2) cells.
    
    1. Finds the row containing red (2) cells.
    2. Creates a green (3) triangle above the red cells, expanding one column to the right for each row up.
    3. Creates a blue (1) triangle below the red cells, shrinking one column for each row down.
    4. Keeps the original red (2) cells unchanged.
    
    Returns the original grid if no red cells are found.
    """
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()

    # Find the row with 2s
    anchor_row = next((i for i, row in enumerate(grid.values) if 2 in row), -1)
    if anchor_row == -1:
        return grid  # No 2s found, return the original grid

    # Count the number of 2s and find the starting column
    twos_row = grid.values[anchor_row]
    n = twos_row.count(2)
    start_col = twos_row.index(2)

    # Fill above with 3s
    for i in range(anchor_row):
        row = anchor_row - i - 1
        for col in range(start_col, min(start_col + n + i + 1, width)):
            grid.set_cell(row, col, 3)

    # Fill below with 1s
    for i in range(1, height - anchor_row):
        row = anchor_row + i
        for col in range(start_col, min(start_col + n - i, width)):
            grid.set_cell(row, col, 1)

    return grid
