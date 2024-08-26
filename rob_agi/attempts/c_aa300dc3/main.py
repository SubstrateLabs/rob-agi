from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_aa300dc3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the aa300dc3 challenge by finding a diagonal path through the largest
    connected black region and placing sky blue squares at regular intervals.
    
    1. Identify the largest connected black region
    2. Determine the start and end points for the diagonal
    3. Use A* algorithm to find a path through the black region
    4. Place sky blue squares along the path at regular intervals
    5. Return the modified grid
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find the largest connected black region
    black_regions = input_grid.find_connected_regions(0)
    largest_black_region = max(black_regions, key=len)
    
    # Determine start and end points
    start = min(largest_black_region)
    end = max(largest_black_region)
    
    # Find path using A* algorithm
    path = a_star(input_grid, start, end)
    
    # Place sky blue squares along the path
    step_size = max(len(path) // 8, 1)  # Adjust step size based on path length
    for i in range(0, len(path), step_size):
        r, c = path[i]
        output_grid.set_cell(r, c, 8)  # 8 represents sky blue
    
    return output_grid

def a_star(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    
    def heuristic(a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
        return neighbors
    
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == end:
            return reconstruct_path(came_from, current)
        
        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []
