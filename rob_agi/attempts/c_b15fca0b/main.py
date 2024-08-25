from rob_agi.colored_grid import ColoredGrid

BLACK, BLUE, RED, YELLOW = 0, 1, 2, 4

def solve_b15fca0b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling enclosed areas with yellow (4).
    
    The function identifies areas that are completely surrounded by blue (1) lines,
    red (2) squares, or the grid edges. These enclosed areas are filled with yellow (4).
    Areas that have a path to the edge of the grid (including diagonal paths) remain black (0).
    Blue lines and red squares remain unchanged.
    
    Algorithm:
    1. Perform an initial flood fill from the edges to mark cells with a path to the edge, including diagonal paths.
    2. Fill unmarked black cells with yellow.
    3. Ensure blue lines and red squares remain unchanged.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def is_valid(row, col):
        return 0 <= row < rows and 0 <= col < cols

    def is_edge(row, col):
        return row == 0 or col == 0 or row == rows - 1 or col == cols - 1

    def flood_fill(row, col):
        if not is_valid(row, col) or visited[row][col] or grid.values[row][col] in [1, 2]:
            return
        visited[row][col] = True
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                flood_fill(row + dr, col + dc)

    # Perform edge flood fill
    for r in range(rows):
        for c in range(cols):
            if is_edge(r, c) and grid.values[r][c] == 0:
                flood_fill(r, c)

    # Fill enclosed areas with yellow
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0 and not visited[r][c]:
                grid.values[r][c] = 4

    return grid
