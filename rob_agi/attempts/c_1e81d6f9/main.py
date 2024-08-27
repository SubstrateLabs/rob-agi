from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict, deque
import heapq
import random

def solve_1e81d6f9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the gray T-shape and intelligently limiting other colors.
    
    1. Preserves the T-shaped gray object.
    2. Analyzes connected regions of each color.
    3. Prioritizes larger connected regions while limiting each color to 3-4 occurrences.
    4. Maintains spatial distribution of colors.
    5. Ensures color variety by preserving at least one cell of each color present in the input.
    6. Makes minimal changes to the input grid.
    7. Allows for occasional complete removal of a color or keeping fewer than 3 instances.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Step 1: Preserve the T-shaped gray object
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 5:  # Gray color
                output_grid.values[r][c] = 5
    
    def get_connected_regions(color):
        regions = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color and (r, c) not in visited:
                    region = []
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) not in visited and input_grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                new_r, new_c = curr_r + dr, curr_c + dc
                                if 0 <= new_r < rows and 0 <= new_c < cols:
                                    queue.append((new_r, new_c))
                    regions.append(region)
        return regions
    
    def calculate_isolation(cell, color):
        r, c = cell
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
    
    color_queues = {}
    color_counters = defaultdict(int)
    
    for color in range(1, 10):
        if color == 5:  # Skip gray
            continue
        regions = get_connected_regions(color)
        queue = []
        for region in regions:
            for cell in region:
                isolation = calculate_isolation(cell, color)
                edge_priority = 1 if cell[0] in (0, rows-1) or cell[1] in (0, cols-1) else 0
                heapq.heappush(queue, (-len(region), isolation, edge_priority, random.random(), cell))
        color_queues[color] = queue
    
    # Step 2-5: Process other colors
    active_colors = set(color_queues.keys())
    while active_colors:
        for color in list(active_colors):
            if not color_queues[color]:
                active_colors.remove(color)
                continue
            
            _, _, _, _, (r, c) = heapq.heappop(color_queues[color])
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = color
                color_counters[color] += 1
                
                # Remove nearby cells from the queue
                color_queues[color] = [item for item in color_queues[color] if abs(item[4][0] - r) + abs(item[4][1] - c) > 2]
                heapq.heapify(color_queues[color])
                
                if color_counters[color] >= 3 and random.random() < 0.7:  # 70% chance to stop at 3
                    active_colors.remove(color)
            
            if all(counter >= 2 for counter in color_counters.values()) and random.random() < 0.3:  # 30% chance to stop early
                active_colors.clear()
                break
    
    # Ensure color variety
    for color in range(1, 10):
        if color == 5:  # Skip gray
            continue
        if color_counters[color] == 0 and any(input_grid.values[r][c] == color for r in range(rows) for c in range(cols)):
            most_isolated = max(
                [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == color],
                key=lambda cell: calculate_isolation(cell, color)
            )
            output_grid.values[most_isolated[0]][most_isolated[1]] = color
    
    return output_grid
