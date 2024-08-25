from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions.
    2. For rows with yellow squares at both ends:
       - Replaces squares between yellows with alternating magenta (6) and black (0).
    3. For columns with yellow squares:
       - Fills spaces between topmost and bottommost yellows with alternating orange (7) and sky blue (8).
    4. Ensures connectivity between yellow squares by filling paths with alternating orange and sky blue.
    5. Adjusts the grid to prevent new colors (7 or 8) from touching horizontally or vertically.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find yellow squares
    yellow_positions = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 4]

    # Process rows with yellow squares at both ends
    for r in range(rows):
        row_yellows = [c for c in range(cols) if output_grid.values[r][c] == 4]
        if len(row_yellows) == 2:
            for c in range(row_yellows[0] + 1, row_yellows[1]):
                output_grid.values[r][c] = 6 if (c - row_yellows[0]) % 2 else 0

    # Process columns with yellow squares
    for c in range(cols):
        col_yellows = [r for r in range(rows) if output_grid.values[r][c] == 4]
        if len(col_yellows) >= 2:
            for r in range(col_yellows[0] + 1, col_yellows[-1]):
                if output_grid.values[r][c] not in [4, 6, 0]:
                    output_grid.values[r][c] = 8 if (r - col_yellows[0]) % 2 else 7

    # Connect yellow squares
    def dfs(r: int, c: int, visited: List[List[bool]], path: List[Tuple[int, int]]):
        if output_grid.values[r][c] == 4:
            fill_path(path)
            return True
        visited[r][c] = True
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and output_grid.values[nr][nc] != 6:
                if dfs(nr, nc, visited, path + [(nr, nc)]):
                    return True
        return False

    def fill_path(path: List[Tuple[int, int]]):
        for i, (r, c) in enumerate(path):
            if output_grid.values[r][c] not in [4, 6, 0]:
                output_grid.values[r][c] = 8 if i % 2 else 7

    for start_r, start_c in yellow_positions:
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        dfs(start_r, start_c, visited, [])

    # Final pass to separate new colors
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] in [7, 8]:
                neighbors = [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols]
                if any(output_grid.values[nr][nc] in [7, 8] for nr, nc in neighbors):
                    output_grid.values[r][c] = 6

    return output_grid
