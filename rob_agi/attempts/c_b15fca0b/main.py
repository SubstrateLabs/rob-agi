from rob_agi.colored_grid import ColoredGrid

BLACK, BLUE, RED, YELLOW = 0, 1, 2, 4

def solve_b15fca0b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling enclosed areas with yellow (4).
    
    The function identifies areas that are completely surrounded by blue (1) lines,
    red (2) squares, or the grid edges. These enclosed areas are filled with yellow (4).
    Areas that have a path to the edge of the grid remain black (0).
    Blue lines and red squares remain unchanged.
    
    Algorithm:
    1. Perform an initial flood fill from the edges to mark cells with a path to the edge.
    2. Fill unmarked black cells with yellow using a flood fill algorithm.
    3. Ensure blue lines and red squares remain unchanged.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    marked = [[False for _ in range(cols)] for _ in range(rows)]

    def is_valid_cell(row, col):
        return 0 <= row < rows and 0 <= col < cols

    def is_boundary_cell(row, col):
        return row == 0 or col == 0 or row == rows - 1 or col == cols - 1

    def edge_flood_fill(row, col):
        if not is_valid_cell(row, col) or marked[row][col] or grid.values[row][col] in [BLUE, RED]:
            return
        
        marked[row][col] = True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            edge_flood_fill(row + dr, col + dc)

    def yellow_flood_fill(row, col):
        if not is_valid_cell(row, col) or marked[row][col] or grid.values[row][col] in [BLUE, RED, YELLOW]:
            return
        
        grid.values[row][col] = YELLOW
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            yellow_flood_fill(row + dr, col + dc)

    # Edge flood fill
    for r in range(rows):
        for c in range(cols):
            if is_boundary_cell(r, c) and grid.values[r][c] == BLACK:
                edge_flood_fill(r, c)

    # Yellow flood fill
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == BLACK and not marked[r][c]:
                yellow_flood_fill(r, c)

    return grid
