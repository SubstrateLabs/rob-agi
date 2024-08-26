from rob_agi.colored_grid import ColoredGrid
from collections import deque, defaultdict
import heapq

def solve_c35c1b4c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c35c1b4c challenge by expanding the largest contiguous region of color.
    
    The function identifies the largest connected region of a single color,
    then expands this region based on an expansion potential map. The expansion
    respects other significant color regions, maintains the overall structure,
    and adapts to different input scenarios.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the largest region expanded.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify the largest contiguous region
    largest_region = []
    expanding_color = None
    for color in range(10):
        regions = input_grid.find_connected_regions(color)
        if regions and len(regions[0]) > len(largest_region):
            largest_region = regions[0]
            expanding_color = color
    
    # Step 2: Create an expansion potential map
    potential_map = [[0 for _ in range(cols)] for _ in range(rows)]
    
    def calculate_potential(x, y):
        if (x, y) in largest_region:
            return 0
        distance = min(abs(x-rx) + abs(y-ry) for rx, ry in largest_region)
        base_potential = 100 / (distance + 1)
        if input_grid.values[x][y] == 0:  # Higher potential for black cells
            base_potential *= 1.5
        return base_potential
    
    for x in range(rows):
        for y in range(cols):
            potential_map[x][y] = calculate_potential(x, y)
    
    # Step 3: Identify significant secondary regions
    significant_regions = set()
    threshold = rows * cols * 0.1
    for color in range(10):
        if color != expanding_color:
            regions = input_grid.find_connected_regions(color)
            for region in regions:
                if len(region) >= threshold:
                    significant_regions.update(region)
    
    # Step 4: Expansion process
    grid = input_grid.deep_copy()
    cells_to_expand = [(potential_map[x][y], x, y) for x in range(rows) for y in range(cols) if (x, y) not in largest_region]
    heapq.heapify(cells_to_expand)
    
    expansion_limit = rows * cols * 0.7
    while cells_to_expand and len(largest_region) < expansion_limit:
        potential, x, y = heapq.heappop(cells_to_expand)
        
        if (x, y) in significant_regions and potential < 80:
            continue
        
        if potential < 30:  # Lower threshold for expansion
            break
        
        grid.values[x][y] = expanding_color
        largest_region.append((x, y))
        
        # Update potential for adjacent cells
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in largest_region:
                new_potential = calculate_potential(nx, ny)
                potential_map[nx][ny] = new_potential
                heapq.heappush(cells_to_expand, (new_potential, nx, ny))
    
    # Step 5: Smoothing pass
    for x in range(rows):
        for y in range(cols):
            if grid.values[x][y] != expanding_color:
                neighbors = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= x+dx < rows and 0 <= y+dy < cols and grid.values[x+dx][y+dy] == expanding_color)
                if neighbors >= 3:
                    grid.values[x][y] = expanding_color
    
    return grid
