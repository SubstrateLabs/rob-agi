from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_aa300dc3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the aa300dc3 challenge by creating a diagonal line of 8 sky blue squares
    through the black region of the grid, avoiding obstacles.

    1. Analyze the grid to find the best starting corner
    2. Use A* algorithm to find a diagonal path of 8 steps
    3. Place 8 sky blue squares along the found path
    4. If no valid path is found, try alternative approaches
    5. Return the modified grid or the original if no solution is found
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    def find_best_start():
        corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
        best_corner = max(corners, key=lambda c: sum(1 for r in range(rows) for col in range(cols) 
                                                     if input_grid.get_cell(r, col) == 0 and 
                                                     abs(r-c[0]) == abs(col-c[1])))
        return min([(r, c) for r in range(rows) for c in range(cols)
                    if input_grid.get_cell(r, c) == 0],
                   key=lambda pos: abs(pos[0]-best_corner[0]) + abs(pos[1]-best_corner[1]))

    start = find_best_start()
    end = (rows-1-start[0], cols-1-start[1])  # Opposite corner

    def heuristic(a, b):
        return max(abs(b[0] - a[0]), abs(b[1] - a[1]))  # Diagonal distance

    def get_neighbors(pos):
        r, c = pos
        neighbors = []
        for dr, dc in [(1,1), (1,-1), (-1,1), (-1,-1), (0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.get_cell(nr, nc) == 0:
                neighbors.append((nr, nc))
        return neighbors

    path = a_star(input_grid, start, end)

    if len(path) < 8:
        # If path is too short, try to extend it
        while len(path) < 8 and path:
            last = path[-1]
            for neighbor in get_neighbors(last):
                if neighbor not in path:
                    path.append(neighbor)
                    break
            else:
                break
    elif len(path) > 8:
        # If path is too long, truncate it
        path = path[:8]

    if len(path) == 8:
        for r, c in path:
            output_grid.set_cell(r, c, 8)
        return output_grid
    else:
        return input_grid  # Unable to find a solution, return original grid

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
