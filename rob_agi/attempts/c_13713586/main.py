from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from typing import List, Tuple
from collections import deque

def solve_13713586(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions in all directions,
    while preserving gray boundaries and respecting the "first to reach" rule.
    
    The algorithm works as follows:
    1. Create a copy of the input grid.
    2. Identify all colored positions (excluding black and gray).
    3. Sort colored positions from top to bottom, then left to right.
    4. For each colored position, perform a flood fill in all directions.
    5. Preserve gray boundaries by restoring them after expansion.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_colored_positions(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        positions = []
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] not in [0, 5]:
                    positions.append((r, c, grid.values[r][c]))
        return sorted(positions)

    def flood_fill(grid: List[List[int]], row: int, col: int, color: int):
        queue = deque([(row, col)])
        while queue:
            r, c = queue.popleft()
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
                grid[r][c] = color
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    queue.append((r + dr, c + dc))

    colored_positions = get_colored_positions(grid)

    for r, c, color in colored_positions:
        flood_fill(grid.values, r, c, color)

    # Preserve gray boundaries
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 5:
                grid.values[r][c] = 5

    return grid
