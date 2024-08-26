from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_aa300dc3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the aa300dc3 challenge by creating a diagonal line of 8 sky blue squares
    through the black region of the grid, avoiding obstacles.

    1. Find the starting point (black cell nearest to a corner)
    2. Determine the diagonal direction
    3. Calculate initial step size
    4. Place 8 sky blue squares along the diagonal, adjusting for obstacles
    5. If unable to place all 8 squares, adjust step size and try again
    6. Return the modified grid or the original if no solution is found
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    # Find starting point
    corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    start = min([(r, c) for r in range(rows) for c in range(cols)
                if input_grid.get_cell(r, c) == 0],
                key=lambda pos: min(abs(pos[0]-cr) + abs(pos[1]-cc) for cr, cc in corners))

    # Determine diagonal direction
    direction = (1, 1) if start[0] + start[1] < rows - 1 + cols - 1 else (1, -1)

    # Calculate initial step size
    initial_step_size = max(min(rows, cols) // 7, 1)
    step_size = initial_step_size

    def place_sky_blue_squares():
        squares_placed = 0
        current_position = start
        for _ in range(8):
            r, c = current_position
            if 0 <= r < rows and 0 <= c < cols and output_grid.get_cell(r, c) == 0:
                output_grid.set_cell(r, c, 8)
                squares_placed += 1
                next_position = (r + direction[0] * step_size, c + direction[1] * step_size)
                if not (0 <= next_position[0] < rows and 0 <= next_position[1] < cols and output_grid.get_cell(*next_position) == 0):
                    # Search for a nearby black cell
                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and output_grid.get_cell(nr, nc) == 0:
                            next_position = (nr, nc)
                            break
                    else:
                        return squares_placed
                current_position = next_position
            else:
                return squares_placed
        return squares_placed

    while True:
        squares_placed = place_sky_blue_squares()
        if squares_placed == 8:
            return output_grid
        output_grid = input_grid.deep_copy()  # Reset the grid
        if squares_placed < 8:
            step_size -= 1
        else:
            step_size += 1
        if step_size < 1:
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
