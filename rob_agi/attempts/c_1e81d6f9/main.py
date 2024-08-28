from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict, deque
import heapq
import random

def solve_1e81d6f9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the gray T-shape and intelligently limiting other colors.
    
    1. Preserves the T-shaped gray object.
    2. Analyzes connected regions and isolation of each colored cell.
    3. Prioritizes keeping isolated cells and significant clusters.
    4. Limits each color to generally 3-4 occurrences, with some flexibility.
    5. Maintains spatial distribution across 3x3 sectors of the grid.
    6. Ensures color variety by preserving at least one cell of each input color.
    7. Applies a small randomization factor for occasional color removal or extra preservation.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    def calculate_isolation(r, c, color):
        isolation = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    if input_grid.values[new_r][new_c] == color:
                        isolation -= 1
                    else:
                        isolation += 1
        return isolation
    
    def get_connected_regions(color):
        regions = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color and (r, c) not in visited:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and input_grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                new_r, new_c = curr_r + dr, curr_c + dc
                                if 0 <= new_r < rows and 0 <= new_c < cols:
                                    stack.append((new_r, new_c))
                    regions.append(region)
        return regions
    
    # Step 1: Preserve the T-shaped gray object
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 5:  # Gray color
                output_grid.values[r][c] = 5
    
    color_data = {}
    for color in range(1, 10):
        if color == 5:  # Skip gray
            continue
        regions = get_connected_regions(color)
        cells = [(r, c) for region in regions for r, c in region]
        isolations = [calculate_isolation(r, c, color) for r, c in cells]
        color_data[color] = {
            'regions': regions,
            'cells': cells,
            'isolations': isolations
        }
    
    # Step 2-3: Preserve isolated cells and significant clusters
    for color, data in color_data.items():
        cells = data['cells']
        isolations = data['isolations']
        if cells:
            # Always keep the most isolated cell
            most_isolated = max(range(len(cells)), key=lambda i: isolations[i])
            r, c = cells[most_isolated]
            output_grid.values[r][c] = color
            
            # Keep 1-2 cells from the largest cluster
            largest_cluster = max(data['regions'], key=len)
            if len(largest_cluster) > 1:
                r, c = random.choice(largest_cluster)
                if (r, c) != cells[most_isolated]:
                    output_grid.values[r][c] = color
    
    # Step 4-5: Maintain spatial distribution and limit color occurrences
    sector_size = rows // 3
    for color, data in color_data.items():
        color_count = sum(row.count(color) for row in output_grid.values)
        if color_count >= 4:
            continue
        
        for sector_r in range(3):
            for sector_c in range(3):
                sector_cells = [
                    (r, c) for r, c in data['cells']
                    if sector_r * sector_size <= r < (sector_r + 1) * sector_size
                    and sector_c * sector_size <= c < (sector_c + 1) * sector_size
                    and output_grid.values[r][c] == 0
                ]
                if sector_cells and color_count < 4:
                    r, c = max(sector_cells, key=lambda cell: calculate_isolation(*cell, color))
                    output_grid.values[r][c] = color
                    color_count += 1
                if color_count >= 4:
                    break
            if color_count >= 4:
                break
    
    # Step 6: Ensure color variety
    for color in range(1, 10):
        if color == 5:  # Skip gray
            continue
        if not any(color in row for row in output_grid.values) and color in color_data:
            cells = color_data[color]['cells']
            if cells:
                r, c = max(cells, key=lambda cell: calculate_isolation(*cell, color))
                output_grid.values[r][c] = color
    
    # Step 7: Apply randomization factor
    if random.random() < 0.1:  # 10% chance
        colors = [c for c in range(1, 10) if c != 5 and any(c in row for row in output_grid.values)]
        if colors:
            color = random.choice(colors)
            if sum(row.count(color) for row in output_grid.values) <= 3:
                # Remove the color
                for r in range(rows):
                    for c in range(cols):
                        if output_grid.values[r][c] == color:
                            output_grid.values[r][c] = 0
            else:
                # Allow an extra cell
                cells = [(r, c) for r in range(rows) for c in range(cols) 
                         if input_grid.values[r][c] == color and output_grid.values[r][c] == 0]
                if cells:
                    r, c = random.choice(cells)
                    output_grid.values[r][c] = color
    
    return output_grid
