from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import heapq

def solve_14754a24(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by connecting yellow squares with L-shaped red paths.
    
    1. Identifies yellow (4) squares in the grid.
    2. Finds potential connections between yellow squares within a 3x3 area.
    3. Processes connections, creating L-shaped red (2) paths between yellow squares.
    4. Prioritizes efficient connections, leaving some yellows unchanged if inefficient.
    5. Converts remaining yellow squares to red if adjacent to a red square.
    
    Returns a new ColoredGrid with the transformed values.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def find_yellow_squares() -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 4]
    
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def find_potential_connections(yellows: List[Tuple[int, int]]) -> List[Tuple[int, Tuple[int, int], Tuple[int, int]]]:
        connections = []
        for i, (r1, c1) in enumerate(yellows):
            for r2, c2 in yellows[i+1:]:
                dist = manhattan_distance((r1, c1), (r2, c2))
                if dist <= 3:
                    heapq.heappush(connections, (dist, (r1, c1), (r2, c2)))
        return connections
    
    def generate_l_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        r1, c1 = start
        r2, c2 = end
        return [(r, c1) for r in range(min(r1, r2), max(r1, r2) + 1)] + \
               [(r2, c) for c in range(min(c1, c2), max(c1, c2) + 1) if c != c1]
    
    def is_valid_path(path: List[Tuple[int, int]]) -> bool:
        return all(grid.values[r][c] in [4, 5] for r, c in path if (r, c) != path[0] and (r, c) != path[-1])
    
    def apply_path(path: List[Tuple[int, int]]) -> None:
        for r, c in path:
            grid.values[r][c] = 2
    
    yellows = find_yellow_squares()
    connections = find_potential_connections(yellows)
    processed = set()
    
    while connections:
        _, start, end = heapq.heappop(connections)
        if start in processed or end in processed:
            continue
        
        path = generate_l_path(start, end)
        if is_valid_path(path):
            apply_path(path)
            processed.add(start)
            processed.add(end)
    
    # Convert remaining adjacent yellows to red
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 4:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 2:
                        grid.values[r][c] = 2
                        break
    
    return grid
