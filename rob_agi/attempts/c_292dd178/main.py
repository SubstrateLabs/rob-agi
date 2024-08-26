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
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with enclosed areas filled with red.
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
            if not visited[row][col] and grid.values[row][col] != 1:
                grid.values[row][col] = 2

    return grid
