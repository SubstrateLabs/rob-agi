from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_696d4842(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating the longest possible continuous lines for each color,
    with a preference for vertical connections. The process involves:
    1. Finding all cells of each unique color.
    2. Creating vertical connections from the topmost to bottommost cell of each color.
    3. Creating horizontal connections in each row between the leftmost and rightmost cells.
    4. Filling in gaps surrounded by the same color.
    5. Removing isolated cells of each color.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    colors = set(cell for row in output_grid.values for cell in row if cell != 0)

    for color in colors:
        cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == color]
        if len(cells) <= 1:
            continue

        top = min(r for r, _ in cells)
        bottom = max(r for r, _ in cells)
        left = min(c for _, c in cells)
        right = max(c for _, c in cells)

        # Create vertical connection
        for r in range(top, bottom + 1):
            if any(output_grid.values[r][c] == color for c in range(cols)):
                output_grid.values[r][left] = color

        # Create horizontal connections
        for r in range(top, bottom + 1):
            if any(output_grid.values[r][c] == color for c in range(cols)):
                for c in range(left, right + 1):
                    output_grid.values[r][c] = color

    # Fill gaps and remove isolated cells
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] != 0:
                neighbors = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and
                                output_grid.values[r + dr][c + dc] == output_grid.values[r][c])
                if neighbors == 0:
                    output_grid.values[r][c] = 0

    return output_grid
