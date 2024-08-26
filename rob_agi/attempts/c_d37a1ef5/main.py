from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_d37a1ef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding the red frame inwards using a flood fill algorithm,
    while preserving other colored cells and their adjacent black cells.
    
    The function:
    1. Identifies the original red frame
    2. Copies all cells from the input grid
    3. Uses a flood fill algorithm to expand the red frame inwards, stopping at non-black cells
    4. Returns a new grid with the expanded red frame
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()

    # Find original frame boundaries
    top = next(r for r in range(rows) if 2 in input_grid.values[r])
    bottom = next(r for r in range(rows-1, -1, -1) if 2 in input_grid.values[r])
    left = next(c for c in range(cols) if any(row[c] == 2 for row in input_grid.values))
    right = next(c for c in range(cols-1, -1, -1) if any(row[c] == 2 for row in input_grid.values))

    # Implement flood fill algorithm
    queue = deque()
    # Add original frame cells to queue
    for r in range(top, bottom + 1):
        queue.append((r, left))
        queue.append((r, right))
    for c in range(left + 1, right):
        queue.append((top, c))
        queue.append((bottom, c))

    while queue:
        r, c = queue.popleft()
        if (top <= r <= bottom and left <= c <= right and
            input_grid.values[r][c] == 0 and new_grid.values[r][c] == 0):
            new_grid.values[r][c] = 2
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))

    return new_grid
