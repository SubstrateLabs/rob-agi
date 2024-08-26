from rob_agi.colored_grid import ColoredGrid

def solve_90347967(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving non-black cells to the top-right corner and rotating them 90 degrees clockwise.
    
    1. Collects all non-black cells from the input grid, scanning left to right and bottom to top.
    2. Creates a new grid with the same dimensions as the input, filled with black (0) cells.
    3. Places the collected non-black cells in the top-right corner of the new grid, starting from the rightmost column
       and moving left, giving the appearance of a 90-degree clockwise rotation.
    4. Any cells that would fall outside the grid bounds are omitted.
    """
    rows, cols = input_grid.get_dimensions()
    non_black_cells = []

    # Collect non-black cells (left to right, bottom to top)
    for col in range(cols):
        for row in range(rows-1, -1, -1):
            if input_grid.values[row][col] != 0:
                non_black_cells.append(input_grid.values[row][col])

    # Create a new grid filled with black cells
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    # Place non-black cells in the top-right corner
    for i, value in enumerate(non_black_cells):
        new_col = cols - 1 - (i % rows)
        new_row = i // rows
        if new_col >= 0 and new_row < rows:
            new_grid[new_row][new_col] = value

    return ColoredGrid(values=new_grid)
