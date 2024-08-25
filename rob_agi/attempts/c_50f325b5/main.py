from rob_agi.colored_grid import ColoredGrid
from collections import deque
import copy

def solve_50f325b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Expand yellow (8) regions in the grid based on specific rules:
    1. Expansion prioritizes right and down directions.
    2. Limited expansion to left and up is allowed.
    3. Expansion is limited based on the original region's size and position.
    4. The expansion process stops when it reaches its limit or can't expand further.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    processed = set()

    def is_valid_position(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def get_expansion_limit(region):
        return min(len(region) * 2, 10)  # Adjust this formula as needed

    def expand_region(start_r, start_c):
        original_region = []
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if (r, c) not in processed and grid.get_cell(r, c) == 8:
                processed.add((r, c))
                original_region.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if is_valid_position(nr, nc) and (nr, nc) not in processed:
                        queue.append((nr, nc))

        expansion_limit = get_expansion_limit(original_region)
        expansion_queue = deque(original_region)
        expanded = 0

        while expansion_queue and expanded < expansion_limit:
            r, c = expansion_queue.popleft()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid_position(nr, nc) and grid.get_cell(nr, nc) != 8:
                    grid.set_cell(nr, nc, 8)
                    expansion_queue.append((nr, nc))
                    expanded += 1
                    if expanded >= expansion_limit:
                        break
            if expanded >= expansion_limit:
                break

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8 and (r, c) not in processed:
                expand_region(r, c)

    return grid
