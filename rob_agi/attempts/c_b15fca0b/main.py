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
    1. Create a deep copy of the input grid.
    2. Initialize a 'reachable' grid to track cells that can be reached from edges or red squares.
    3. Perform flood fill from red squares and non-blue edge cells, including diagonal movements.
    4. Fill unreachable black cells with yellow.
    5. Return the modified grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    reachable = [[False for _ in range(cols)] for _ in range(rows)]

    def flood_fill(r, c):
        if not (0 <= r < rows and 0 <= c < cols) or reachable[r][c] or grid.values[r][c] == BLUE:
            return
        reachable[r][c] = True
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                flood_fill(r + dr, c + dc)

    # Identify starting points and perform flood fill
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == RED or (grid.values[r][c] == BLACK and (r in [0, rows-1] or c in [0, cols-1])):
                flood_fill(r, c)

    # Fill enclosed areas
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == BLACK and not reachable[r][c]:
                grid.values[r][c] = YELLOW

    return grid
