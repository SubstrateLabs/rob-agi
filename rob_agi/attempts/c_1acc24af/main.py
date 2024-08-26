from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_1acc24af(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing gray (5) regions to red (2) if they meet specific criteria:
    1. The region's bounding box is at least 2x2 in size.
    2. The region contains at least 4 connected gray cells.
    3. The region is not a perfect rectangle (cell count != bounding box area).
    The function uses a flood fill algorithm to identify connected gray regions and applies the transformation
    if all criteria are met. Non-qualifying gray regions remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    processed = set()

    def flood_fill(row: int, col: int) -> Tuple[List[Tuple[int, int]], int, int, int, int, int]:
        queue = deque([(row, col)])
        connected_cells = []
        min_row, max_row, min_col, max_col = row, row, col, col
        cell_count = 0

        while queue:
            r, c = queue.popleft()
            if (r, c) not in processed and output_grid.get_cell(r, c) == 5:
                processed.add((r, c))
                connected_cells.append((r, c))
                cell_count += 1
                min_row, max_row = min(min_row, r), max(max_row, r)
                min_col, max_col = min(min_col, c), max(max_col, c)

                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        queue.append((nr, nc))

        return connected_cells, min_row, max_row, min_col, max_col, cell_count

    cells_to_transform = []

    for row in range(rows):
        for col in range(cols):
            if output_grid.get_cell(row, col) == 5 and (row, col) not in processed:
                connected_cells, min_row, max_row, min_col, max_col, cell_count = flood_fill(row, col)
                width = max_col - min_col + 1
                height = max_row - min_row + 1
                bounding_box_area = width * height

                if width >= 2 and height >= 2 and cell_count >= 4 and cell_count != bounding_box_area:
                    cells_to_transform.extend(connected_cells)

    for r, c in cells_to_transform:
        output_grid.set_cell(r, c, 2)

    return output_grid
