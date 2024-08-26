from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_54db823b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving all colored regions that are connected
    to the edge of the grid, and removing (setting to black) all isolated regions.

    1. Creates a new grid initialized with all black squares.
    2. Uses a flood fill algorithm starting from all edges of the input grid.
    3. During flood fill, copies the color of each reachable cell to the new grid.
    4. Returns the new grid where only regions connected to the edges are preserved.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def flood_fill(r: int, c: int):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or input_grid.values[r][c] == 0:
            return
        visited[r][c] = True
        output_grid.values[r][c] = input_grid.values[r][c]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            flood_fill(r + dr, c + dc)

    # Flood fill from all edges
    for c in range(cols):
        flood_fill(0, c)  # Top edge
        flood_fill(rows-1, c)  # Bottom edge
    for r in range(rows):
        flood_fill(r, 0)  # Left edge
        flood_fill(r, cols-1)  # Right edge

    return output_grid
