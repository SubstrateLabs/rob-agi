from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
import heapq

def solve_14754a24(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by connecting yellow squares with L-shaped red paths.
    
    1. Identifies yellow (4) squares and their clusters in the grid.
    2. Preserves yellow clusters of 3 or more squares.
    3. Creates a priority queue for potential connections between remaining yellow squares.
    4. Processes connections, creating L-shaped or alternative red (2) paths between yellow squares.
    5. Converts remaining isolated yellow squares to red if adjacent to a red square or part of a small cluster.
    6. Optimizes the solution by removing unnecessary red squares and ensuring path continuity.
    
    Returns a new ColoredGrid with the transformed values.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def find_yellow_squares() -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 4]
    
    def find_yellow_clusters() -> List[Set[Tuple[int, int]]]:
        yellows = set(find_yellow_squares())
        clusters = []
        while yellows:
            start = yellows.pop()
            cluster = {start}
            stack = [start]
            while stack:
                r, c = stack.pop()
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in yellows:
                        yellows.remove((nr, nc))
                        cluster.add((nr, nc))
                        stack.append((nr, nc))
            clusters.append(cluster)
        return clusters
    
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def find_potential_connections(yellows: List[Tuple[int, int]]) -> List[Tuple[int, Tuple[int, int], Tuple[int, int]]]:
        connections = []
        for i, p1 in enumerate(yellows):
            for p2 in yellows[i+1:]:
                dist = manhattan_distance(p1, p2)
                if dist <= 5:  # Increased range for more flexibility
                    heapq.heappush(connections, (dist, p1, p2))
        return connections
    
    def generate_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        r1, c1 = start
        r2, c2 = end
        path = []
        if abs(r1 - r2) <= abs(c1 - c2):  # Horizontal L or straight line
            path = [(r1, c) for c in range(min(c1, c2), max(c1, c2) + 1)] + \
                   [(r, c2) for r in range(min(r1, r2), max(r1, r2) + 1) if r != r1]
        else:  # Vertical L or straight line
            path = [(r, c1) for r in range(min(r1, r2), max(r1, r2) + 1)] + \
                   [(r2, c) for c in range(min(c1, c2), max(c1, c2) + 1) if c != c1]
        return [p for p in path if p != start and p != end]  # Exclude start and end points
    
    def is_valid_path(path: List[Tuple[int, int]]) -> bool:
        return all(0 <= r < rows and 0 <= c < cols and grid.values[r][c] in [0, 4, 5] for r, c in path)
    
    def apply_path(path: List[Tuple[int, int]]) -> None:
        for r, c in path:
            grid.values[r][c] = 2
    
    yellow_clusters = find_yellow_clusters()
    preserve = set()
    for cluster in yellow_clusters:
        if len(cluster) >= 3:
            preserve.update(cluster)
    
    yellows = [y for y in find_yellow_squares() if y not in preserve]
    connections = find_potential_connections(yellows)
    processed = set()
    
    while connections:
        _, start, end = heapq.heappop(connections)
        if start in processed or end in processed:
            continue
        
        path = generate_path(start, end)
        if is_valid_path(path):
            apply_path(path)
            grid.values[start[0]][start[1]] = 2
            grid.values[end[0]][end[1]] = 2
            processed.add(start)
            processed.add(end)
    
    # Convert remaining yellows to red if they're part of a small cluster or adjacent to red
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 4 and (r, c) not in preserve:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 2:
                        grid.values[r][c] = 2
                        break
    
    # Optimize by removing unnecessary red squares and ensuring path continuity
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 2:
                neighbors = [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols]
                red_neighbors = sum(1 for nr, nc in neighbors if grid.values[nr][nc] == 2)
                if red_neighbors == 0:
                    grid.values[r][c] = 0  # Convert isolated red to black
                elif red_neighbors == 1 and (r, c) not in yellows:
                    grid.values[r][c] = 0  # Convert end points to black if they weren't original yellows
    
    return grid
