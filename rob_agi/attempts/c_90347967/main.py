from rob_agi.colored_grid import ColoredGrid

def solve_90347967(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving non-black cells to the top-right corner and rotating them 90 degrees clockwise.
    
    1. Collects all non-black cells from the input grid.
    2. Sorts the collected cells based on their original position (left to right, bottom to top).
    3. Creates a new grid with the same dimensions as the input.
    4. Places the sorted non-black cells in the top-right corner of the new grid, rotated 90 degrees clockwise.
    5. Fills the rest of the new grid with black (0) cells.
    """
    rows, cols = input_grid.get_dimensions()
    non_black_cells = []

    # Collect non-black cells
    for col in range(cols):
        for row in range(rows-1, -1, -1):
            if input_grid.values[row][col] != 0:
                non_black_cells.append(input_grid.values[row][col])

    # Create a new grid filled with black cells
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Place non-black cells in the top-right corner, rotated 90 degrees clockwise
    for i, value in enumerate(non_black_cells):
        new_row = i // cols
        new_col = cols - 1 - (i % cols)
        if new_row < rows:
            new_grid[new_row][new_col] = value

    return ColoredGrid(values=new_grid)
