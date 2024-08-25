from rob_agi.colored_grid import ColoredGrid
from collections import deque
import random

def solve_50f325b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Expand yellow (8) regions in the grid based on the following rules:
    1. Identify all yellow regions in the input grid.
    2. Calculate expansion energy for each region based on its size and grid dimensions.
    3. Implement a probabilistic expansion process, favoring cells adjacent to yellow regions.
    4. Expand regions based on their energy and surrounding space.
    5. Perform final smoothing to ensure connectivity and remove isolated cells.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    processed = set()
    random.seed(42)  # For reproducibility

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
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in processed:
                        queue.append((nr, nc))
        return region

    # Identify yellow regions
    yellow_regions = []
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8 and (r, c) not in processed:
                yellow_regions.append(get_region(r, c))

    # Calculate expansion energy
    grid_area = rows * cols
    energies = [min(len(region) * 0.5, grid_area * 0.1) for region in yellow_regions]

    # Expansion process
    candidates = set()
    for region in yellow_regions:
        for r, c in region:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 8:
                    candidates.add((nr, nc))

    while candidates and any(energy > 0 for energy in energies):
        r, c = candidates.pop()
        if grid.get_cell(r, c) == 8:
            continue

        # Calculate expansion probability
        yellow_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                               if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 8)
        nearest_region_index = min(range(len(yellow_regions)), 
                                   key=lambda i: min((r-yr)**2 + (c-yc)**2 for yr, yc in yellow_regions[i]))
        prob = min(1.0, energies[nearest_region_index] / 100 + yellow_neighbors * 0.2)

        if random.random() < prob:
            grid.set_cell(r, c, 8)
            energies[nearest_region_index] -= 1
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 8:
                    candidates.add((nr, nc))

    # Connectivity and smoothing
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                yellow_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                       if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 8)
                if yellow_neighbors == 0:
                    grid.set_cell(r, c, 0)  # Remove isolated yellow cells
            else:
                yellow_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                       if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 8)
                if yellow_neighbors >= 3:
                    grid.set_cell(r, c, 8)  # Convert to yellow if surrounded

    return grid
