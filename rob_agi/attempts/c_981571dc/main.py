from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_981571dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling black areas with colors.
    
    The function starts from the edges of black areas and moves inward, replacing black (0) cells
    with appropriate colors based on their non-black neighbors. This process continues
    until no black cells remain. The algorithm uses a queue-based approach to ensure
    that the fill progresses from the edges inward, maintaining the continuity of existing patterns.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with black areas filled.
    """
    def get_neighbors(row, col):
        return [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]

    def is_valid(row, col):
        return 0 <= row < rows and 0 <= col < cols

    def get_first_non_black_neighbor(row, col):
        for nr, nc in get_neighbors(row, col):
            if is_valid(nr, nc) and grid[nr][nc] != 0:
                return grid[nr][nc]
        return None

    grid = input_grid.deep_copy()
    rows, cols = len(grid.values), len(grid.values[0])
    queue = deque()

    # Identify edge cells
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0 and any(is_valid(nr, nc) and grid.values[nr][nc] != 0 for nr, nc in get_neighbors(r, c)):
                queue.append((r, c))

    # Fill process
    while queue:
        r, c = queue.popleft()
        if grid.values[r][c] == 0:
            color = get_first_non_black_neighbor(r, c)
            if color is not None:
                grid.values[r][c] = color
                for nr, nc in get_neighbors(r, c):
                    if is_valid(nr, nc) and grid.values[nr][nc] == 0:
                        queue.append((nr, nc))

    # Final check for any remaining black cells
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                color = get_first_non_black_neighbor(r, c)
                if color is not None:
                    grid.values[r][c] = color

    return grid
