from rob_agi.colored_grid import ColoredGrid

def solve_dbc1a6ce(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the dbc1a6ce challenge by connecting blue squares (1) with sky-colored lines (8).
    
    The solution involves:
    1. Finding all blue squares (1) in the grid.
    2. For each blue square, extending lines in all four directions until another blue square or the grid boundary is reached.
    3. Marking the extended lines with sky color (8).
    4. Cleaning up any sky-colored squares that are not part of a valid connection.
    
    Args:
    input_grid (ColoredGrid): The input grid to be solved.
    
    Returns:
    ColoredGrid: The solved grid with connected blue squares.
    """
    def find_next_one(grid, row, col, dr, dc):
        r, c = row + dr, col + dc
        while 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            if grid[r][c] == 1:
                return r, c
            r, c = r + dr, c + dc
        return None

    def mark_connection(grid, start_row, start_col, end_row, end_col):
        dr = (end_row - start_row) // max(abs(end_row - start_row), 1)
        dc = (end_col - start_col) // max(abs(end_col - start_col), 1)
        r, c = start_row + dr, start_col + dc
        while (r, c) != (end_row, end_col):
            grid[r][c] = 8
            r, c = r + dr, c + dc

    def is_valid_connection(grid, row, col, visited=None):
        if visited is None:
            visited = set()
        if (row, col) in visited or grid[row][col] not in [1, 8]:
            return False
        if grid[row][col] == 1:
            return True
        visited.add((row, col))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                if is_valid_connection(grid, nr, nc, visited):
                    return True
        return False

    result = input_grid.deep_copy()
    grid = result.values

    # First pass: Mark potential connections
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    next_one = find_next_one(grid, r, c, dr, dc)
                    if next_one:
                        mark_connection(grid, r, c, next_one[0], next_one[1])

    # Cleanup pass: Remove invalid connections
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 8 and not is_valid_connection(grid, r, c):
                grid[r][c] = 0

    return result
