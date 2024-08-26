from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_54db823b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving colored regions that are:
    1. Connected to the edge of the grid.
    2. Adjacent (above or to the left) to already preserved regions.
    All other isolated regions are removed (set to black).

    The algorithm works in two passes:
    1. Preserves all edge-connected regions.
    2. Processes remaining regions, preserving those adjacent to preserved ones.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def is_edge(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def has_preserved_neighbor(r: int, c: int) -> bool:
        for dr, dc in [(-1, 0), (0, -1)]:  # Check above and left
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and output_grid.values[nr][nc] != 0:
                return True
        return False

    def flood_fill(r: int, c: int, color: int, preserve: bool):
        if not is_valid(r, c) or visited[r][c] or input_grid.values[r][c] != color:
            return
        visited[r][c] = True
        if preserve:
            output_grid.values[r][c] = color
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            flood_fill(r + dr, c + dc, color, preserve)

    # First pass: preserve edge-connected regions
    for r in range(rows):
        for c in range(cols):
            if is_edge(r, c) and not visited[r][c] and input_grid.values[r][c] != 0:
                flood_fill(r, c, input_grid.values[r][c], True)

    # Second pass: process remaining regions
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and input_grid.values[r][c] != 0:
                preserve = has_preserved_neighbor(r, c)
                flood_fill(r, c, input_grid.values[r][c], preserve)

    return output_grid
