from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_aa300dc3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the aa300dc3 challenge by creating a path of 8 sky blue squares
    through the black region of the grid, avoiding obstacles.

    1. Identify potential starting points adjacent to the border
    2. For each starting point, use a modified pathfinding algorithm to find paths of exactly 8 steps
    3. Prioritize diagonal moves but allow occasional horizontal or vertical moves
    4. Evaluate paths based on diagonality, proximity to border, and obstacle avoidance
    5. Place 8 sky blue squares along the best found path
    6. Return the modified grid or the original if no valid path is found
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    def find_potential_starts():
        starts = []
        for r in range(rows):
            for c in range(cols):
                if input_grid.get_cell(r, c) == 0 and (r == 0 or r == rows-1 or c == 0 or c == cols-1):
                    starts.append((r, c))
        return starts

    def is_border(r, c):
        return r == 0 or r == rows-1 or c == 0 or c == cols-1

    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        for dr, dc in [(1,1), (1,-1), (-1,1), (-1,-1), (0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.get_cell(nr, nc) == 0:
                neighbors.append((nr, nc))
        return neighbors

    def evaluate_path(path):
        diagonality = sum(1 for i in range(len(path)-1) if abs(path[i][0]-path[i+1][0]) == abs(path[i][1]-path[i+1][1]))
        border_proximity = 2 if is_border(path[-1][0], path[-1][1]) else 0
        return diagonality + border_proximity

    def dfs(start, path, visited):
        if len(path) == 8:
            return [path] if is_border(path[-1][0], path[-1][1]) else []
        
        paths = []
        for neighbor in get_neighbors(path[-1]):
            if neighbor not in visited:
                new_path = path + [neighbor]
                new_visited = visited | {neighbor}
                paths.extend(dfs(start, new_path, new_visited))
        return paths

    best_path = None
    best_score = -1

    for start in find_potential_starts():
        paths = dfs(start, [start], {start})
        for path in paths:
            score = evaluate_path(path)
            if score > best_score:
                best_path = path
                best_score = score

    if best_path:
        for r, c in best_path:
            output_grid.set_cell(r, c, 8)
        return output_grid
    else:
        return input_grid  # Unable to find a solution, return original grid

def modified_a_star(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    
    def heuristic(a, b):
        return max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    
    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        for dr, dc in [(1,1), (1,-1), (-1,1), (-1,-1), (0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
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
        
        if current == end and g_score[current] == 7:  # Exactly 8 steps (including start)
            return reconstruct_path(came_from, current)
        
        if g_score[current] >= 7:  # Don't explore further if we've already taken 8 steps
            continue
        
        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []

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
