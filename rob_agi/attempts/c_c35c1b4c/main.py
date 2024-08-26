from rob_agi.colored_grid import ColoredGrid
from collections import deque, defaultdict
import heapq

def solve_c35c1b4c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c35c1b4c challenge by enhancing the most significant shape in the grid.
    
    This function identifies the largest contiguous shape, creates an enhancement map,
    and carefully expands the shape while respecting other significant structures.
    It uses a priority-based approach to fill gaps, smooth edges, and expand in a
    controlled manner, maintaining the overall balance of the grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the most significant shape enhanced.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify the largest contiguous shape
    regions = {}
    for color in range(10):
        regions[color] = input_grid.find_connected_regions(color)
    
    largest_region = max((region for color_regions in regions.values() for region in color_regions), key=len)
    expanding_color = input_grid.values[largest_region[0][0]][largest_region[0][1]]
    
    # Step 2: Create enhancement map
    enhancement_map = [[0 for _ in range(cols)] for _ in range(rows)]
    
    def calculate_enhancement_priority(x, y):
        if (x, y) in largest_region:
            return 0
        adjacent = any((x+dx, y+dy) in largest_region for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)])
        if adjacent:
            return 3  # High priority for adjacent cells
        distance = min(abs(x-rx) + abs(y-ry) for rx, ry in largest_region)
        if distance <= 2:
            return 2  # Medium priority for nearby cells
        return 1  # Low priority for other cells
    
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
    cells_to_enhance = [(enhancement_map[x][y], x, y) for x in range(rows) for y in range(cols) if enhancement_map[x][y] > 0]
    cells_to_enhance.sort(reverse=True)
    
    for priority, x, y in cells_to_enhance:
        if (x, y) in protected_regions:
            continue
        
        neighbors = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        if 0 <= x+dx < rows and 0 <= y+dy < cols and grid.values[x+dx][y+dy] == expanding_color)
        
        if neighbors >= 2 or (neighbors == 1 and priority == 3):
            grid.values[x][y] = expanding_color
    
    # Step 5: Smoothing pass
    for x in range(rows):
        for y in range(cols):
            if grid.values[x][y] != expanding_color:
                neighbors = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= x+dx < rows and 0 <= y+dy < cols and grid.values[x+dx][y+dy] == expanding_color)
                if neighbors >= 3:
                    grid.values[x][y] = expanding_color
    
    return grid
