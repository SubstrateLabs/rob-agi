from rob_agi.colored_grid import ColoredGrid

def solve_981571dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling black areas with colors.
    
    The function performs the following steps:
    1. Create a deep copy of the input grid.
    2. Iterate through the grid, filling black cells based on neighbors.
    3. Handle the top-left corner and first row/column separately.
    4. Perform a final check to ensure no black cells remain.
    
    This approach ensures consistent pattern extension, prioritizing left-to-right
    and top-to-bottom filling, while handling edge cases and ensuring complete filling.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with black areas filled.
    """
    grid = input_grid.deep_copy()
    rows, cols = len(grid.values), len(grid.values[0])

    # Main filling process
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                if c > 0:
                    grid.values[r][c] = grid.values[r][c-1]
                elif r > 0:
                    grid.values[r][c] = grid.values[r-1][c]

    # Handle top-left corner
    if grid.values[0][0] == 0:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != 0:
                    grid.values[0][0] = grid.values[r][c]
                    break
            if grid.values[0][0] != 0:
                break

    # Handle first row and column
    for c in range(1, cols):
        if grid.values[0][c] == 0:
            grid.values[0][c] = grid.values[0][c-1]
    for r in range(1, rows):
        if grid.values[r][0] == 0:
            grid.values[r][0] = grid.values[r-1][0]

    # Final check
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                neighbors = []
                if c > 0:
                    neighbors.append(grid.values[r][c-1])
                if r > 0:
                    neighbors.append(grid.values[r-1][c])
                if c < cols - 1:
                    neighbors.append(grid.values[r][c+1])
                if r < rows - 1:
                    neighbors.append(grid.values[r+1][c])
                if neighbors:
                    grid.values[r][c] = next(filter(lambda x: x != 0, neighbors), 1)

    return grid
