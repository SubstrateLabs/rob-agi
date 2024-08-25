from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d2acf2cb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies yellow (4) squares and their positions.
    2. For columns with yellow squares:
       - Fills spaces between topmost and bottommost yellows with alternating orange (7) and sky blue (8).
    3. For rows with yellow squares at both ends:
       - Replaces squares between yellows with alternating magenta (6) and black (0).
    4. Ensures connectivity between yellow squares by filling paths with alternating orange and sky blue.
    5. Adjusts the grid to prevent new colors (7 or 8) from touching horizontally or vertically.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Clean up any existing orange or sky blue squares
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] in [7, 8]:
                output_grid.values[r][c] = 0

    # Find yellow squares
    yellow_positions = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 4]

    # Process columns with yellow squares
    for c in range(cols):
        col_yellows = [r for r in range(rows) if output_grid.values[r][c] == 4]
        if len(col_yellows) >= 2:
            for r in range(col_yellows[0] + 1, col_yellows[-1]):
                output_grid.values[r][c] = 7 if (r - col_yellows[0]) % 2 else 8

    # Process rows with yellow squares at both ends
    for r in range(rows):
        row_yellows = [c for c in range(cols) if output_grid.values[r][c] == 4]
        if len(row_yellows) == 2:
            for c in range(row_yellows[0] + 1, row_yellows[1]):
                output_grid.values[r][c] = 6 if (c - row_yellows[0]) % 2 else 0

    # Connect remaining unconnected yellow squares
    def bfs(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        queue = [(start, [start])]
        visited = set()
        while queue:
            (r, c), path = queue.pop(0)
            if (r, c) == end:
                return path
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    for i, start in enumerate(yellow_positions):
        for end in yellow_positions[i+1:]:
            if start[0] == end[0] or start[1] == end[1]:
                continue  # Already connected in same row or column
            path = bfs(start, end)
            for j, (r, c) in enumerate(path[1:-1], 1):
                if output_grid.values[r][c] not in [4, 6, 0]:
                    output_grid.values[r][c] = 8 if j % 2 else 7

    # Adjust for color adjacency
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] in [7, 8]:
                neighbors = [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols]
                if any(output_grid.values[nr][nc] in [7, 8] for nr, nc in neighbors):
                    output_grid.values[r][c] = 6

    return output_grid
