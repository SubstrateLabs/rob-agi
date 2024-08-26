from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_292dd178(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling enclosed spaces with red (2).
    
    The function performs the following steps:
    1. Creates a deep copy of the input grid.
    2. Uses a flood fill algorithm to mark all reachable cells from the edges.
    3. Fills all unreachable cells (except blue ones) with red (2).
    4. Preserves all blue (1) cells and cells reachable from non-blue edges.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with enclosed areas filled with red.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # Create a boolean matrix to mark reachable cells
    reachable = [[False for _ in range(cols)] for _ in range(rows)]
    
    # Helper function for flood fill
    def flood_fill(start_r: int, start_c: int):
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] != 1 and not reachable[r][c]:
                reachable[r][c] = True
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    queue.append((r + dr, c + dc))
    
    # Start flood fill from edge cells that are not blue (1) or red (2)
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid.values[r][c] not in [1, 2]:
                flood_fill(r, c)
    
    # Fill unreachable cells with red (2)
    for r in range(rows):
        for c in range(cols):
            if not reachable[r][c] and grid.values[r][c] != 1:
                grid.values[r][c] = 2
    
    return grid
