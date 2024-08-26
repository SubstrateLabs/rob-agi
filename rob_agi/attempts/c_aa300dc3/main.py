from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_aa300dc3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the aa300dc3 challenge by creating a path of 8 sky blue squares
    through the black region of the grid, avoiding obstacles.

    1. Identify potential starting points adjacent to the border
    2. For each starting point, use a depth-first search to find paths of exactly 8 steps
    3. Prioritize diagonal moves but allow occasional horizontal or vertical moves
    4. Evaluate paths based on diagonality, proximity to border, and obstacle avoidance
    5. Place 8 sky blue squares along the best found path
    6. Return the modified grid or the original if no valid path is found
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()

    def is_border(r, c):
        return r == 0 or r == rows-1 or c == 0 or c == cols-1

    def get_neighbors(r, c):
        neighbors = []
        for dr, dc in [(1,1), (1,-1), (-1,1), (-1,-1), (0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.get_cell(nr, nc) == 0:
                neighbors.append((nr, nc))
        return neighbors

    def evaluate_path(path):
        diagonality = sum(1 for i in range(len(path)-1) if abs(path[i][0]-path[i+1][0]) == abs(path[i][1]-path[i+1][1]))
        border_proximity = sum(1 for r, c in path if is_border(r, c))
        return diagonality + border_proximity

    def dfs(r, c, path, visited):
        if len(path) == 8:
            return [path] if is_border(r, c) else []
        
        paths = []
        for nr, nc in get_neighbors(r, c):
            if (nr, nc) not in visited:
                new_path = path + [(nr, nc)]
                new_visited = visited | {(nr, nc)}
                paths.extend(dfs(nr, nc, new_path, new_visited))
        return paths

    best_path = None
    best_score = -1

    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 0 and is_border(r, c):
                paths = dfs(r, c, [(r, c)], {(r, c)})
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
