from rob_agi.colored_grid import ColoredGrid

def solve_981571dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling black areas with colors.
    
    The function performs the following steps:
    1. Create a deep copy of the input grid.
    2. Iterate through each cell from left to right, top to bottom.
    3. If a black cell (value 0) is found:
       a. If it's in the first column, set its value to the cell above (if exists).
       b. Otherwise, set its value to the cell to its left.
    4. After processing all rows, fill any remaining black cells in the first row.
    
    This approach ensures consistent pattern extension, prioritizing left-to-right filling.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with black areas filled.
    """
    grid = input_grid.deep_copy()
    rows, cols = len(grid.values), len(grid.values[0])

    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                if c == 0 and r > 0:
                    grid.values[r][c] = grid.values[r-1][c]
                elif c > 0:
                    grid.values[r][c] = grid.values[r][c-1]

    # Handle remaining black cells in the first row
    for c in range(1, cols):
        if grid.values[0][c] == 0:
            grid.values[0][c] = grid.values[0][c-1]

    return grid
