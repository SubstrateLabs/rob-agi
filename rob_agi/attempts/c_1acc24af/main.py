from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from collections import deque

def solve_1acc24af(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing rectangular gray (5) regions with both dimensions 2 or greater to red (2).
    The function uses a flood fill algorithm to identify connected gray regions,
    checks if they are rectangular and have the required dimensions, then transforms them if conditions are met.
    All other cells, including non-rectangular gray regions, remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    visited = set()

    def flood_fill(row: int, col: int) -> Tuple[Set[Tuple[int, int]], int, int, int, int]:
        region = set()
        queue = deque([(row, col)])
        min_row, max_row, min_col, max_col = row, row, col, col

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or output_grid.get_cell(r, c) != 5:
                continue

            visited.add((r, c))
            region.add((r, c))
            min_row, max_row = min(min_row, r), max(max_row, r)
            min_col, max_col = min(min_col, c), max(max_col, c)

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                queue.append((r + dr, c + dc))

        return region, min_row, max_row, min_col, max_col

    for row in range(rows):
        for col in range(cols):
            if output_grid.get_cell(row, col) == 5 and (row, col) not in visited:
                region, min_row, max_row, min_col, max_col = flood_fill(row, col)
                width = max_col - min_col + 1
                height = max_row - min_row + 1
                is_rectangular = len(region) == width * height

                if is_rectangular and width >= 2 and height >= 2:
                    for r, c in region:
                        output_grid.set_cell(r, c, 2)

    return output_grid
