from rob_agi.colored_grid import ColoredGrid
from collections import deque, defaultdict
import heapq

def solve_c35c1b4c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c35c1b4c challenge by enhancing the most significant shape in the grid.
    
    This function identifies the most significant shape based on size, position, and regularity,
    then enhances it by filling gaps, smoothing edges, and expanding in a controlled manner.
    The enhancement respects other significant structures in the grid and maintains overall balance.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the most significant shape enhanced.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Analyze the grid and identify the most significant shape
    regions = {}
    for color in range(10):
        regions[color] = input_grid.find_connected_regions(color)
    
    def shape_significance(region):
        size = len(region)
        center_x = sum(x for x, y in region) / size
        center_y = sum(y for x, y in region) / size
        centrality = 1 - (abs(center_x - rows/2) / (rows/2) + abs(center_y - cols/2) / (cols/2)) / 2
        compactness = size / ((max(x for x, y in region) - min(x for x, y in region) + 1) *
                              (max(y for x, y in region) - min(y for x, y in region) + 1))
        return size * centrality * compactness

    significant_shape = max(
        ((color, region) for color, color_regions in regions.items() for region in color_regions),
        key=lambda x: shape_significance(x[1])
    )
    expanding_color, largest_region = significant_shape

    # Step 2: Create an enhancement map
    enhancement_map = [[0 for _ in range(cols)] for _ in range(rows)]
    
    def calculate_enhancement_priority(x, y):
        if (x, y) in largest_region:
            return 0
        distance = min(abs(x-rx) + abs(y-ry) for rx, ry in largest_region)
        priority = 100 / (distance + 1)
        if input_grid.values[x][y] == 0:  # Higher priority for black cells
            priority *= 1.5
        return priority
    
    for x in range(rows):
        for y in range(cols):
            enhancement_map[x][y] = calculate_enhancement_priority(x, y)
    
    # Step 3: Identify protected areas
    protected_regions = set()
    threshold = rows * cols * 0.05
    for color, color_regions in regions.items():
        if color != expanding_color:
            for region in color_regions:
                if len(region) >= threshold:
                    protected_regions.update(region)
    
    # Step 4: Enhancement process
    grid = input_grid.deep_copy()
    cells_to_enhance = [(enhancement_map[x][y], x, y) for x in range(rows) for y in range(cols) if (x, y) not in largest_region]
    heapq.heapify(cells_to_enhance)
    
    enhancement_limit = min(rows * cols * 0.7, len(largest_region) * 1.5)
    while cells_to_enhance and len(largest_region) < enhancement_limit:
        priority, x, y = heapq.heappop(cells_to_enhance)
        
        if (x, y) in protected_regions and priority < 80:
            continue
        
        if priority < 20:  # Lower threshold for enhancement
            break
        
        grid.values[x][y] = expanding_color
        largest_region.append((x, y))
        
        # Update priority for adjacent cells
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in largest_region:
                new_priority = calculate_enhancement_priority(nx, ny)
                enhancement_map[nx][ny] = new_priority
                heapq.heappush(cells_to_enhance, (new_priority, nx, ny))
    
    # Step 5: Smoothing pass
    for x in range(rows):
        for y in range(cols):
            if grid.values[x][y] != expanding_color:
                neighbors = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= x+dx < rows and 0 <= y+dy < cols and grid.values[x+dx][y+dy] == expanding_color)
                if neighbors >= 3:
                    grid.values[x][y] = expanding_color
    
    return grid
