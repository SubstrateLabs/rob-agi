from rob_agi.colored_grid import ColoredGrid
from collections import deque
import random

def solve_f0df5ff0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a blue (1) path that connects black (0) spaces
    while avoiding other colored squares. The path may branch and form loops to create
    a complex network structure.

    1. Start from existing blue squares or edges of large black areas.
    2. Expand the blue path using a combination of BFS and DFS.
    3. Allow branching and loop formation for more complex structures.
    4. Ensure the blue path is continuous and covers a significant portion of the grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def get_neighbors(r, c):
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]
    
    def find_start_points():
        starts = []
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 1:  # Existing blue squares
                    starts.append((r, c))
        if not starts:  # If no blue squares, start from edges of large black areas
            for r in [0, rows-1]:
                for c in range(cols):
                    if output_grid.get_cell(r, c) == 0:
                        starts.append((r, c))
            for r in range(rows):
                for c in [0, cols-1]:
                    if output_grid.get_cell(r, c) == 0:
                        starts.append((r, c))
        return starts or [(rows//2, cols//2)]  # Fallback to center if no suitable starts
    
    frontier = deque(find_start_points())
    visited = set(frontier)
    
    while frontier:
        r, c = frontier.popleft()
        output_grid.set_cell(r, c, 1)  # Set to blue
        
        neighbors = get_neighbors(r, c)
        random.shuffle(neighbors)  # Randomize expansion direction
        
        for nr, nc in neighbors:
            if (nr, nc) not in visited:
                cell_value = output_grid.get_cell(nr, nc)
                if cell_value == 0:  # Expand to black cells
                    frontier.append((nr, nc))
                    visited.add((nr, nc))
                elif cell_value == 1 and random.random() < 0.1:  # Occasional looping
                    frontier.append((nr, nc))
        
        if random.random() < 0.1:  # Occasional branching
            frontier.extend(n for n in neighbors if n not in visited and output_grid.get_cell(n[0], n[1]) == 0)
    
    return output_grid
