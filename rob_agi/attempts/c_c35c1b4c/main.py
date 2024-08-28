from rob_agi.colored_grid import ColoredGrid
from collections import deque, defaultdict
import math

def solve_c35c1b4c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c35c1b4c challenge by carefully enhancing the most significant shape in the grid.
    
    This function identifies the largest contiguous shape, creates an enhancement map,
    and conservatively expands the shape while respecting other significant structures.
    It uses a priority-based approach to fill gaps and smooth edges, maintaining the
    overall balance and structure of the grid. The enhancement process is adaptive,
    considering the initial shape characteristics and stopping when certain thresholds
    are reached.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the most significant shape subtly enhanced.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Analyze the input grid
    color_frequencies = defaultdict(int)
    for row in input_grid.values:
        for cell in row:
            color_frequencies[cell] += 1
    
    # Step 2: Identify the largest contiguous shape
    regions = {}
    for color in range(10):
        regions[color] = input_grid.find_connected_regions(color)
    
    largest_region = max((region for color_regions in regions.values() for region in color_regions), key=len)
    expanding_color = input_grid.values[largest_region[0][0]][largest_region[0][1]]
    
    # Step 3: Create significance map
    significant_regions = set()
    threshold = rows * cols * 0.05
    for color, color_regions in regions.items():
        for region in color_regions:
            if len(region) >= threshold:
                significant_regions.update(region)
    
    # Step 4: Generate expansion priority map
    def calculate_priority(x, y):
        if (x, y) in largest_region:
            return 0
        neighbors = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        if 0 <= x+dx < rows and 0 <= y+dy < cols and (x+dx, y+dy) in largest_region)
        if neighbors >= 2:
            return 3
        elif neighbors == 1:
            return 2
        distance = min(abs(x-rx) + abs(y-ry) for rx, ry in largest_region)
        return 1 if distance <= 2 else 0
    
    priority_map = [[calculate_priority(x, y) for y in range(cols)] for x in range(rows)]
    
    # Step 5: Calculate initial shape characteristics
    perimeter = sum(1 for x, y in largest_region
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    if (x+dx, y+dy) not in largest_region)
    compactness = 4 * math.pi * len(largest_region) / (perimeter ** 2)
    
    # Step 6: Iterative expansion process
    grid = input_grid.deep_copy()
    original_size = len(largest_region)
    expansion_limit = original_size * 1.2
    
    for priority in [3, 2, 1]:
        cells_to_expand = [(x, y) for x in range(rows) for y in range(cols) if priority_map[x][y] == priority]
        for x, y in cells_to_expand:
            if (x, y) in significant_regions or len(largest_region) >= expansion_limit:
                continue
            neighbors = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                            if 0 <= x+dx < rows and 0 <= y+dy < cols and grid.values[x+dx][y+dy] == expanding_color)
            if neighbors >= 2 or (neighbors == 1 and priority == 3):
                grid.values[x][y] = expanding_color
                largest_region.add((x, y))
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        priority_map[nx][ny] = max(priority_map[nx][ny], calculate_priority(nx, ny))
    
    # Step 7: Smoothing pass
    for x in range(rows):
        for y in range(cols):
            if grid.values[x][y] != expanding_color and (x, y) not in significant_regions:
                neighbors = sum(1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= x+dx < rows and 0 <= y+dy < cols and grid.values[x+dx][y+dy] == expanding_color)
                if neighbors >= 3:
                    grid.values[x][y] = expanding_color
    
    # Step 8: Connectivity check
    final_region = set()
    stack = [next(iter(largest_region))]
    while stack:
        x, y = stack.pop()
        if (x, y) not in final_region and grid.values[x][y] == expanding_color:
            final_region.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    stack.append((nx, ny))
    
    for x in range(rows):
        for y in range(cols):
            if grid.values[x][y] == expanding_color and (x, y) not in final_region:
                grid.values[x][y] = input_grid.values[x][y]
    
    # Step 9: Balance check
    new_frequency = sum(1 for row in grid.values for cell in row if cell == expanding_color)
    if new_frequency > color_frequencies[expanding_color] * 1.25:
        cells_to_revert = sorted(
            [(x, y) for x in range(rows) for y in range(cols) if grid.values[x][y] == expanding_color],
            key=lambda c: priority_map[c[0]][c[1]]
        )
        for x, y in cells_to_revert:
            if new_frequency <= color_frequencies[expanding_color] * 1.25:
                break
            if (x, y) not in largest_region:
                grid.values[x][y] = input_grid.values[x][y]
                new_frequency -= 1
    
    return grid
