from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1acc24af(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing gray (5) regions larger than 2x2 to red (2).
    The function uses a flood fill algorithm to identify connected gray regions
    and transforms them if their size is greater than 4 cells.
    Blue (1) structures and smaller gray regions remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    visited = set()

    def flood_fill(row: int, col: int, color: int) -> Tuple[List[Tuple[int, int]], int]:
        region = []
        size = 0
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < rows and 0 <= c < cols and output_grid.get_cell(r, c) == color and (r, c) not in visited:
                region.append((r, c))
                size += 1
                visited.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        return region, size

    for row in range(rows):
        for col in range(cols):
            if output_grid.get_cell(row, col) == 5 and (row, col) not in visited:
                region, size = flood_fill(row, col, 5)
                if size > 4:
                    for r, c in region:
                        output_grid.set_cell(r, c, 2)

    return output_grid
