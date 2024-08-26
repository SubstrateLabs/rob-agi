from rob_agi.colored_grid import ColoredGrid

def solve_292dd178(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling enclosed spaces with red (2).
    
    The function performs the following steps:
    1. Creates a deep copy of the input grid.
    2. Uses a recursive flood fill algorithm to mark all reachable cells from the grid edges.
    3. Fills all unreachable cells (except blue ones) with red (2).
    4. Preserves all blue (1) cells and cells reachable from the edges.
    5. Handles cases where enclosed areas extend to the grid edges.
    6. Fills areas that are only partially enclosed by blue cells.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with enclosed and partially enclosed areas filled with red.
    """
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()
    visited = [[False for _ in range(width)] for _ in range(height)]

    def flood_fill(row: int, col: int):
        if row < 0 or row >= height or col < 0 or col >= width:
            return
        if grid.values[row][col] == 1 or visited[row][col]:
            return
        visited[row][col] = True
        flood_fill(row-1, col)
        flood_fill(row+1, col)
        flood_fill(row, col-1)
        flood_fill(row, col+1)

    # Start flood fill from all edges
    for r in range(height):
        flood_fill(r, 0)
        flood_fill(r, width-1)
    for c in range(width):
        flood_fill(0, c)
        flood_fill(height-1, c)

    # Fill unvisited, non-blue cells with red
    for row in range(height):
        for col in range(width):
            if grid.values[row][col] != 1:  # If not blue
                if not visited[row][col]:  # If not reachable from edges
                    grid.values[row][col] = 2  # Fill with red
                elif grid.values[row][col] == 8:  # If sky blue
                    # Check if it's partially enclosed by blue
                    if (row > 0 and grid.values[row-1][col] == 1) or \
                       (row < height-1 and grid.values[row+1][col] == 1) or \
                       (col > 0 and grid.values[row][col-1] == 1) or \
                       (col < width-1 and grid.values[row][col+1] == 1):
                        grid.values[row][col] = 2  # Fill with red if partially enclosed

    return grid
