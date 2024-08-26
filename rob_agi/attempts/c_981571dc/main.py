from rob_agi.colored_grid import ColoredGrid

def solve_981571dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling black areas with colors.
    
    The function performs the following steps:
    1. Create a deep copy of the input grid.
    2. Perform a left-to-right fill for each row.
    3. Perform a top-to-bottom fill for each column.
    4. Handle edge cases for the leftmost column and top row.
    
    This approach ensures that patterns are extended consistently,
    prioritizing horizontal continuations followed by vertical continuations.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with black areas filled.
    """
    grid = input_grid.deep_copy()
    rows, cols = len(grid.values), len(grid.values[0])

    # Left-to-right fill
    for r in range(rows):
        for c in range(1, cols):
            if grid.values[r][c] == 0 and grid.values[r][c-1] != 0:
                grid.values[r][c] = grid.values[r][c-1]

    # Top-to-bottom fill
    for c in range(cols):
        for r in range(1, rows):
            if grid.values[r][c] == 0 and grid.values[r-1][c] != 0:
                grid.values[r][c] = grid.values[r-1][c]

    # Handle edge cases
    for r in range(1, rows):
        if grid.values[r][0] == 0:
            grid.values[r][0] = grid.values[r-1][0]
    
    for c in range(1, cols):
        if grid.values[0][c] == 0:
            grid.values[0][c] = grid.values[0][c-1]

    return grid
