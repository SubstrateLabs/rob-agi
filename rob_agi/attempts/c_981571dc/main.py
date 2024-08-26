from rob_agi.colored_grid import ColoredGrid

def solve_981571dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling black areas with colors.
    
    The function performs three passes over the grid:
    1. Fill from left: Extends patterns from left to right.
    2. Fill from top: Extends patterns from top to bottom.
    3. Final fill: Fills any remaining black cells using left or top neighbors.
    
    This approach ensures that patterns are extended in a consistent manner,
    maintaining the "downward and rightward" flow of color patterns.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with black areas filled.
    """
    grid = input_grid.deep_copy()
    rows, cols = len(grid.values), len(grid.values[0])

    # First Pass - Fill from Left
    for r in range(rows):
        for c in range(1, cols):
            if grid.values[r][c] == 0 and grid.values[r][c-1] != 0:
                grid.values[r][c] = grid.values[r][c-1]

    # Second Pass - Fill from Top
    for c in range(cols):
        for r in range(1, rows):
            if grid.values[r][c] == 0 and grid.values[r-1][c] != 0:
                grid.values[r][c] = grid.values[r-1][c]

    # Final Pass - Fill Remaining
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                if c > 0 and grid.values[r][c-1] != 0:
                    grid.values[r][c] = grid.values[r][c-1]
                elif r > 0:
                    grid.values[r][c] = grid.values[r-1][c]

    return grid
