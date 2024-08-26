from rob_agi.colored_grid import ColoredGrid
from collections import deque, defaultdict
import heapq

def solve_c35c1b4c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c35c1b4c challenge by expanding the largest contiguous region of color using a color pressure model.
    
    The function identifies the largest connected region of a single color,
    then expands this region based on a pressure map. The expansion respects
    other significant color regions and uses a dynamic pressure threshold.
    
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
    
    # Step 2: Create a pressure map
    pressure_map = [[0 for _ in range(cols)] for _ in range(rows)]
    for x, y in largest_region:
        pressure_map[x][y] = 100
    
    # Calculate pressure for other cells
    def calculate_pressure(x, y):
        if (x, y) in largest_region:
            return 100
        queue = deque([(x, y, 0)])
        visited = set()
        while queue:
            cx, cy, dist = queue.popleft()
            if (cx, cy) in largest_region:
                return 100 / (dist + 1)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    queue.append((nx, ny, dist + 1))
        return 0
    
    for x in range(rows):
        for y in range(cols):
            if (x, y) not in largest_region:
                pressure_map[x][y] = calculate_pressure(x, y)
    
    # Step 3: Identify other significant regions
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
    cells_to_expand = [(pressure_map[x][y], x, y) for x in range(rows) for y in range(cols) if (x, y) not in largest_region]
    heapq.heapify(cells_to_expand)
    
    while cells_to_expand:
        pressure, x, y = heapq.heappop(cells_to_expand)
        if pressure < 50:  # Pressure threshold
            break
        
        if (x, y) in significant_regions:
            if pressure < 80:  # Higher threshold for significant regions
                continue
        
        grid.values[x][y] = expanding_color
        largest_region.append((x, y))
        
        # Update pressure for adjacent cells
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in largest_region:
                new_pressure = calculate_pressure(nx, ny)
                pressure_map[nx][ny] = new_pressure
                heapq.heappush(cells_to_expand, (new_pressure, nx, ny))
        
        # Termination condition
        if len(largest_region) >= rows * cols * 0.7:
            break
    
    # Step 5: Final pass to fill small gaps
    for x in range(rows):
        for y in range(cols):
            if grid.values[x][y] != expanding_color:
                neighbors = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                                if 0 <= x+dx < rows and 0 <= y+dy < cols and grid.values[x+dx][y+dy] == expanding_color)
                if neighbors >= 5:
                    grid.values[x][y] = expanding_color
    
    return grid
