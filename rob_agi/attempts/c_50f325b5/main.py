from rob_agi.colored_grid import ColoredGrid
from collections import deque
import random

def solve_50f325b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Expand yellow (8) regions in the grid based on the following rules:
    1. Identify all yellow regions in the input grid.
    2. For each region, calculate an expansion factor based on its size and grid dimensions.
    3. Create an expansion boundary around each region, larger on the right and bottom.
    4. Implement a probabilistic expansion within the boundary, favoring right and down directions.
    5. Resolve conflicts between expanding regions.
    6. Ensure connectivity in the final expanded regions.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    processed = set()
    random.seed(42)  # For reproducibility

    def is_valid_position(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def get_expansion_factor(region_size):
        grid_area = rows * cols
        return min(int(region_size * 0.5), int(grid_area * 0.1))

    def get_expansion_boundary(region):
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        return (
            max(0, min_r - height // 2),
            max(0, min_c - width // 2),
            min(rows - 1, max_r + height),
            min(cols - 1, max_c + width * 2)
        )

    def expand_region(region):
        expansion_factor = get_expansion_factor(len(region))
        boundary = get_expansion_boundary(region)
        expansion_queue = deque(sorted(region, key=lambda x: (x[1], x[0])))  # Prioritize right and down
        expanded = 0

        while expansion_queue and expanded < expansion_factor:
            r, c = expansion_queue.popleft()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (is_valid_position(nr, nc) and
                    boundary[0] <= nr <= boundary[2] and
                    boundary[1] <= nc <= boundary[3] and
                    (nr, nc) not in processed and
                    grid.get_cell(nr, nc) != 8):
                    
                    # Probabilistic expansion
                    prob = 0.8 if dr >= 0 and dc >= 0 else 0.2
                    if random.random() < prob:
                        grid.set_cell(nr, nc, 8)
                        processed.add((nr, nc))
                        expansion_queue.append((nr, nc))
                        expanded += 1
                        if expanded >= expansion_factor:
                            break

    def get_region(start_r, start_c):
        region = []
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if (r, c) not in processed and grid.get_cell(r, c) == 8:
                processed.add((r, c))
                region.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if is_valid_position(nr, nc) and (nr, nc) not in processed:
                        queue.append((nr, nc))
        return region

    # Identify and expand all yellow regions
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8 and (r, c) not in processed:
                region = get_region(r, c)
                expand_region(region)

    # Final pass to ensure connectivity
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 8:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if is_valid_position(r+dr, c+dc) and grid.get_cell(r+dr, c+dc) == 8)
                if neighbors >= 3:
                    grid.set_cell(r, c, 8)

    return grid
