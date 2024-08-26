from rob_agi.colored_grid import ColoredGrid
from collections import deque
import random
import math

def solve_50f325b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Expand yellow (8) regions in the grid based on the following rules:
    1. Identify all yellow regions in the input grid.
    2. Calculate expansion energy for each region based on its size and grid dimensions.
    3. Determine directional preference for each region.
    4. Create an expansion probability map for adjacent cells.
    5. Implement a probabilistic expansion process, favoring cells based on the probability map.
    6. Ensure connectivity and apply shape constraints during expansion.
    7. Perform final smoothing to refine the expanded regions.
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

    # Calculate expansion energy and directional preference
    grid_area = rows * cols
    energies = []
    directional_preferences = []
    for region in yellow_regions:
        energy = min(len(region) * 0.75, grid_area * 0.15)
        energies.append(energy)
        
        # Calculate directional preference
        center_r = sum(r for r, _ in region) / len(region)
        center_c = sum(c for _, c in region) / len(region)
        pref_r = sum(r - center_r for r, _ in region)
        pref_c = sum(c - center_c for _, c in region)
        magnitude = math.sqrt(pref_r**2 + pref_c**2)
        if magnitude > 0:
            pref_r, pref_c = pref_r / magnitude, pref_c / magnitude
        directional_preferences.append((pref_r, pref_c))

    def calculate_expansion_probability(r, c, region_index):
        region = yellow_regions[region_index]
        energy = energies[region_index]
        pref_r, pref_c = directional_preferences[region_index]
        
        # Distance factor
        min_dist = min((r-yr)**2 + (c-yc)**2 for yr, yc in region)
        dist_factor = 1 / (1 + min_dist)
        
        # Directional factor
        center_r = sum(yr for yr, _ in region) / len(region)
        center_c = sum(yc for _, yc in region) / len(region)
        dir_r, dir_c = r - center_r, c - center_c
        dir_magnitude = math.sqrt(dir_r**2 + dir_c**2)
        if dir_magnitude > 0:
            dir_r, dir_c = dir_r / dir_magnitude, dir_c / dir_magnitude
        directional_factor = (dir_r * pref_r + dir_c * pref_c + 1) / 2  # Normalize to [0, 1]
        
        # Energy factor
        energy_factor = min(1.0, energy / 50)
        
        # Combine factors
        prob = dist_factor * directional_factor * energy_factor
        return min(1.0, prob * 0.8)  # Cap at 80% to allow for some randomness

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

        nearest_region_index = min(range(len(yellow_regions)), 
                                   key=lambda i: min((r-yr)**2 + (c-yc)**2 for yr, yc in yellow_regions[i]))
        prob = calculate_expansion_probability(r, c, nearest_region_index)

        if random.random() < prob:
            grid.set_cell(r, c, 8)
            energies[nearest_region_index] -= 1
            yellow_regions[nearest_region_index].append((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) != 8:
                    candidates.add((nr, nc))

    # Final smoothing and connectivity check
    changes = True
    while changes:
        changes = False
        for r in range(rows):
            for c in range(cols):
                yellow_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                       if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.get_cell(r+dr, c+dc) == 8)
                if grid.get_cell(r, c) == 8 and yellow_neighbors == 0:
                    grid.set_cell(r, c, 0)  # Remove isolated yellow cells
                    changes = True
                elif grid.get_cell(r, c) != 8 and yellow_neighbors >= 3:
                    grid.set_cell(r, c, 8)  # Convert to yellow if surrounded
                    changes = True

    return grid
