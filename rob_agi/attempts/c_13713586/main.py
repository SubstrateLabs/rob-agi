from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

from collections import deque

def solve_13713586(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions while preserving gray boundaries.
    
    The algorithm works as follows:
    1. Create a copy of the input grid.
    2. Identify all colored positions (excluding black and gray).
    3. Sort colored positions from top to bottom, then left to right.
    4. For each colored position, perform a flood fill expansion.
    5. Preserve gray boundaries by restoring them after expansion.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion rules.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid_position(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_adjacent_positions(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

    def flood_fill(start_r: int, start_c: int, color: int):
        queue = deque([(start_r, start_c)])
        visited = set()

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue

            visited.add((r, c))
            if grid.values[r][c] == 0 or grid.values[r][c] == color:
                grid.values[r][c] = color
                for nr, nc in get_adjacent_positions(r, c):
                    if is_valid_position(nr, nc) and (nr, nc) not in visited:
                        queue.append((nr, nc))

    # Identify colored positions
    colored_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] not in [0, 5]:
                colored_positions.append((r, c, grid.values[r][c]))

    # Sort colored positions
    colored_positions.sort()

    # Expand colors
    for r, c, color in colored_positions:
        if grid.values[r][c] == color:  # Check if not already expanded
            flood_fill(r, c, color)

    # Preserve gray boundaries
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 5:
                grid.values[r][c] = 5

    return grid
