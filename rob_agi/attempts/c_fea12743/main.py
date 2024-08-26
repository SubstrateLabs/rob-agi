from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_fea12743(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the fea12743 challenge by identifying four distinct colored regions,
    determining their order, finding the starting red region, and applying color
    changes clockwise. The solution maintains the original black cells and
    transforms the colored regions according to the pattern:
    - Starting red region remains unchanged
    - Next region clockwise changes to green
    - Next two regions change to sky blue
    """
    # Step 1: Identify the four distinct colored regions
    regions = find_regions(input_grid)
    
    # Step 2: Determine the order of the regions
    ordered_regions = order_regions(regions, input_grid.get_dimensions())
    
    # Step 3: Find the starting point (the red region that remains unchanged)
    start_index = next((i for i, r in enumerate(ordered_regions) if r['color'] == 2), 0)
    
    # Step 4 & 5: Apply the color changes and copy black cells
    new_grid = ColoredGrid(values=[[0 for _ in range(input_grid.get_dimensions()[1])] 
                                   for _ in range(input_grid.get_dimensions()[0])])
    
    new_colors = [2, 3, 8, 8]  # red, green, sky blue, sky blue
    for i in range(4):
        region = ordered_regions[(start_index + i) % 4]
        new_color = new_colors[i]
        for x, y in region['cells']:
            new_grid.values[x][y] = new_color
    
    # Copy black cells
    for x in range(input_grid.get_dimensions()[0]):
        for y in range(input_grid.get_dimensions()[1]):
            if input_grid.values[x][y] == 0:
                new_grid.values[x][y] = 0
    
    return new_grid

def find_regions(grid: ColoredGrid) -> List[Dict]:
    regions = []
    visited = set()
    for x in range(grid.get_dimensions()[0]):
        for y in range(grid.get_dimensions()[1]):
            if (x, y) not in visited and grid.values[x][y] != 0:
                region = flood_fill(grid, x, y, grid.values[x][y])
                regions.append({
                    'color': grid.values[x][y],
                    'cells': region,
                    'centroid': calculate_centroid(region)
                })
                visited.update(region)
    return regions

def flood_fill(grid: ColoredGrid, x: int, y: int, color: int) -> List[Tuple[int, int]]:
    cells = []
    stack = [(x, y)]
    visited = set()
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in visited or grid.values[cx][cy] != color:
            continue
        visited.add((cx, cy))
        cells.append((cx, cy))
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid.get_dimensions()[0] and 0 <= ny < grid.get_dimensions()[1]:
                stack.append((nx, ny))
    return cells

def calculate_centroid(cells: List[Tuple[int, int]]) -> Tuple[float, float]:
    return sum(x for x, _ in cells) / len(cells), sum(y for _, y in cells) / len(cells)

def order_regions(regions: List[Dict], grid_dimensions: Tuple[int, int]) -> List[Dict]:
    center_x, center_y = grid_dimensions[0] / 2, grid_dimensions[1] / 2
    for region in regions:
        cx, cy = region['centroid']
        if cx < center_x and cy < center_y:
            region['position'] = 'top-left'
        elif cx < center_x and cy >= center_y:
            region['position'] = 'bottom-left'
        elif cx >= center_x and cy < center_y:
            region['position'] = 'top-right'
        else:
            region['position'] = 'bottom-right'
    return sorted(regions, key=lambda r: ['top-left', 'top-right', 'bottom-right', 'bottom-left'].index(r['position']))
