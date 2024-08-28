from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import random
import math

def solve_a04b2602(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying blue patterns to green regions.
    
    The transformation follows these steps:
    1. Identify contiguous green (3) regions using flood fill.
    2. Categorize regions as small (<25 cells), medium (25-100 cells), or large (>100 cells).
    3. Create distance maps for each region (distance to red dots and edges).
    4. Generate blue patterns based on region size and distance maps:
       a. For small regions: Create simple blue patterns, filling about half the region.
       b. For medium and large regions: Use probability function based on distances to create organic patterns.
    5. Implement a "vein" system for medium and large regions to maintain green structures.
    6. Apply pattern smoothing to reduce isolated cells and improve appearance.
    7. Preserve original red dots and areas outside green regions.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int, diagonal: bool = False) -> List[Tuple[int, int]]:
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        if diagonal:
            directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
        return [(r+dr, c+dc) for dr, dc in directions if is_valid(r+dr, c+dc)]

    def find_green_regions() -> List[List[Tuple[int, int]]]:
        visited = set()
        regions = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and input_grid.get_cell(r, c) == 3:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and input_grid.get_cell(curr_r, curr_c) == 3:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            stack.extend(get_neighbors(curr_r, curr_c))
                    regions.append(region)
        return regions

    def categorize_region(region: List[Tuple[int, int]]) -> str:
        size = len(region)
        if size < 25:
            return "small"
        elif size < 100:
            return "medium"
        else:
            return "large"

    def create_distance_map(region: List[Tuple[int, int]], red_dots: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[float, float]]:
        distance_map = {}
        max_distance = math.sqrt(rows**2 + cols**2)
        
        for r, c in region:
            # Distance to nearest red dot
            red_distance = min((abs(r-rr) + abs(c-cc) for rr, cc in red_dots), default=max_distance)
            
            # Distance to nearest edge
            edge_distance = min(r, c, rows-1-r, cols-1-c)
            
            distance_map[(r, c)] = (red_distance / max_distance, edge_distance / max(rows, cols))
        
        return distance_map

    def create_blue_pattern(region: List[Tuple[int, int]], category: str, distance_map: Dict[Tuple[int, int], Tuple[float, float]]):
        red_dots = [(r, c) for r, c in region if input_grid.get_cell(r, c) == 2]
        
        if category == "small":
            blue_cells = set()
            target_blue = len(region) // 2  # Fill about half the region with blue
            
            start_points = red_dots if red_dots else [random.choice(region)]
            
            for start_r, start_c in start_points:
                queue = [(start_r, start_c)]
                while queue and len(blue_cells) < target_blue:
                    r, c = queue.pop(0)
                    if (r, c) in region and (r, c) not in blue_cells and input_grid.get_cell(r, c) != 2:
                        output_grid.set_cell(r, c, 1)
                        blue_cells.add((r, c))
                        queue.extend(get_neighbors(r, c))
            
            # Ensure at least one green cell remains
            if len(blue_cells) == len(region):
                r, c = random.choice(list(blue_cells))
                output_grid.set_cell(r, c, 3)
        
        else:  # Medium or Large region
            base_probability = 0.6 if category == "medium" else 0.7
            
            for r, c in region:
                if input_grid.get_cell(r, c) != 2:  # Don't change red dots
                    red_dist, edge_dist = distance_map[(r, c)]
                    prob = base_probability * (1 - red_dist) * (1 + edge_dist)
                    
                    if random.random() < prob:
                        output_grid.set_cell(r, c, 1)

    def create_green_veins(region: List[Tuple[int, int]], category: str):
        if category == "small":
            return
        
        num_veins = 2 if category == "medium" else 4
        for _ in range(num_veins):
            start = random.choice(region)
            end = random.choice(region)
            path = [start]
            current = start
            while current != end:
                neighbors = get_neighbors(*current)
                next_cell = min(neighbors, key=lambda n: ((n[0]-end[0])**2 + (n[1]-end[1])**2))
                path.append(next_cell)
                current = next_cell
            
            for r, c in path:
                if random.random() < 0.7:  # 70% chance to keep the cell green
                    output_grid.set_cell(r, c, 3)

    def smooth_pattern(region: List[Tuple[int, int]]):
        for _ in range(2):  # Apply smoothing twice
            changes = []
            for r, c in region:
                neighbors = get_neighbors(r, c)
                blue_neighbors = sum(1 for nr, nc in neighbors if output_grid.get_cell(nr, nc) == 1)
                green_neighbors = sum(1 for nr, nc in neighbors if output_grid.get_cell(nr, nc) == 3)
                
                if output_grid.get_cell(r, c) == 1 and green_neighbors > 5:
                    changes.append((r, c, 3))
                elif output_grid.get_cell(r, c) == 3 and blue_neighbors > 5:
                    changes.append((r, c, 1))
            
            for r, c, color in changes:
                output_grid.set_cell(r, c, color)

    green_regions = find_green_regions()
    for region in green_regions:
        category = categorize_region(region)
        red_dots = [(r, c) for r, c in region if input_grid.get_cell(r, c) == 2]
        distance_map = create_distance_map(region, red_dots)
        create_blue_pattern(region, category, distance_map)
        create_green_veins(region, category)
        smooth_pattern(region)

    return output_grid
