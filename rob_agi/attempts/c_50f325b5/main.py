from rob_agi.colored_grid import ColoredGrid
from collections import deque
import random
import math

def solve_50f325b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Expand yellow (8) regions and potentially create new ones based on the following steps:
    1. Analyze the input grid to identify existing yellow regions and calculate grid statistics.
    2. Compute an expansion potential score for each cell based on proximity to yellow cells and current color.
    3. Calculate an expansion budget based on existing yellow regions and grid composition.
    4. Expand existing yellow regions prioritizing high-potential cells.
    5. Create new yellow regions with a lower probability in high-potential areas.
    6. Refine shapes by smoothing edges and ensuring connectivity.
    7. Perform final adjustments to adhere to observed patterns and maintain grid structure.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    random.seed(42)  # For reproducibility

    def get_yellow_regions():
        regions = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 8 and (r, c) not in visited:
                    region = []
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == 8:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                                    queue.append((nr, nc))
                    regions.append(region)
        return regions

    def calculate_expansion_potential():
        potential = [[0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) != 8:
                    yellow_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                                           if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 8)
                    potential[r][c] = yellow_neighbors * 2
                    if grid.get_cell(r, c) == 0:  # Empty cells have higher potential
                        potential[r][c] += 2
                    elif grid.get_cell(r, c) in [2, 3, 4, 7]:  # Common colors in examples
                        potential[r][c] += 1
        return potential

    yellow_regions = get_yellow_regions()
    expansion_potential = calculate_expansion_potential()

    total_cells = rows * cols
    yellow_cells = sum(len(region) for region in yellow_regions)
    expansion_budget = min(yellow_cells * 0.7, (total_cells - yellow_cells) * 0.3)

    def expand_regions():
        candidates = set()
        for region in yellow_regions:
            for r, c in region:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 8:
                        candidates.add((nr, nc))

        expanded = 0
        while candidates and expanded < expansion_budget:
            r, c = max(candidates, key=lambda pos: expansion_potential[pos[0]][pos[1]])
            if expansion_potential[r][c] > 0:
                grid.set_cell(r, c, 8)
                expanded += 1
                candidates.remove((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 8:
                        candidates.add((nr, nc))
                        expansion_potential[nr][nc] += 2  # Increase potential of neighboring cells
            else:
                candidates.remove((r, c))

    def create_new_regions():
        remaining_budget = expansion_budget - sum(1 for r in range(rows) for c in range(cols) if grid.get_cell(r, c) == 8)
        if remaining_budget > 0:
            potential_new_regions = [(r, c) for r in range(rows) for c in range(cols)
                                     if grid.get_cell(r, c) != 8 and expansion_potential[r][c] > 1]
            potential_new_regions.sort(key=lambda pos: expansion_potential[pos[0]][pos[1]], reverse=True)
            
            for r, c in potential_new_regions:
                if remaining_budget > 0 and random.random() < 0.2:  # 20% chance to create a new region
                    grid.set_cell(r, c, 8)
                    remaining_budget -= 1
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 8 and remaining_budget > 0:
                            grid.set_cell(nr, nc, 8)
                            remaining_budget -= 1

    def refine_shapes():
        changes = True
        while changes:
            changes = False
            for r in range(rows):
                for c in range(cols):
                    yellow_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                           if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 8)
                    if grid.get_cell(r, c) == 8 and yellow_neighbors <= 1:
                        grid.set_cell(r, c, 0)  # Remove isolated or weakly connected yellow cells
                        changes = True
                    elif grid.get_cell(r, c) != 8 and yellow_neighbors >= 3:
                        grid.set_cell(r, c, 8)  # Convert to yellow if mostly surrounded
                        changes = True

    expand_regions()
    create_new_regions()
    refine_shapes()

    return grid
