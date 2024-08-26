from rob_agi.colored_grid import ColoredGrid
from typing import Set, Tuple, Dict

def solve_929ab4e9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Finding the central connected region of 2's (red)
    2. Determining the bounding box of this region
    3. Creating a pattern map from the surrounding area
    4. Filling the central region with the pattern
    5. Returning the modified grid
    """
    # Find the central region of 2's
    regions = input_grid.find_connected_regions(2)
    central_region = max(regions, key=lambda r: len(r))

    # Calculate bounding box
    min_x = min(x for x, _ in central_region)
    max_x = max(x for x, _ in central_region)
    min_y = min(y for _, y in central_region)
    max_y = max(y for _, y in central_region)

    # Create pattern map
    pattern_map = create_pattern_map(input_grid, central_region, min_x, max_x, min_y, max_y)

    # Fill the central region
    output_grid = input_grid.deep_copy()
    for x, y in central_region:
        rel_x = (x - min_x) / (max_x - min_x) if max_x > min_x else 0
        rel_y = (y - min_y) / (max_y - min_y) if max_y > min_y else 0
        closest_pattern = min(pattern_map.keys(), key=lambda k: ((k[0]-rel_x)**2 + (k[1]-rel_y)**2))
        output_grid.values[x][y] = pattern_map[closest_pattern]

    return output_grid

def create_pattern_map(grid: ColoredGrid, central_region: Set[Tuple[int, int]], 
                       min_x: int, max_x: int, min_y: int, max_y: int) -> Dict[Tuple[float, float], int]:
    pattern_map = {}
    rows, cols = grid.get_dimensions()
    for x in range(rows):
        for y in range(cols):
            if (x, y) not in central_region:
                rel_x = (x - min_x) / (max_x - min_x) if max_x > min_x else 0
                rel_y = (y - min_y) / (max_y - min_y) if max_y > min_y else 0
                pattern_map[(rel_x, rel_y)] = grid.values[x][y]
    return pattern_map
