from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_54db823b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by removing all colored regions that can be isolated
    by a continuous path of black squares connected to the edge of the grid.
    
    1. Identifies all connected regions of non-black squares in the input grid.
    2. For each region, checks if it can be isolated by black squares.
    3. If a region can be isolated, it is removed (set to black) in the output grid.
    4. If a region cannot be isolated, it is kept as-is in the output grid.
    5. Returns the new grid with isolated regions removed.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    visited = set()

    def flood_fill(r: int, c: int, color: int, region: List[Tuple[int, int]]) -> None:
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or input_grid.values[r][c] != color:
            return
        visited.add((r, c))
        region.append((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            flood_fill(r + dr, c + dc, color, region)

    def can_isolate(region: List[Tuple[int, int]]) -> bool:
        temp_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        for r, c in region:
            temp_grid[r][c] = 1
        
        edge_queue = [(r, c) for r in range(rows) for c in range(cols) 
                      if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and temp_grid[r][c] == 0]
        
        while edge_queue:
            r, c = edge_queue.pop(0)
            if temp_grid[r][c] != 0:
                continue
            temp_grid[r][c] = 2
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and temp_grid[nr][nc] == 0:
                    edge_queue.append((nr, nc))
        
        return any(temp_grid[r][c] == 1 for r, c in region)

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and (r, c) not in visited:
                region = []
                flood_fill(r, c, input_grid.values[r][c], region)
                if not can_isolate(region):
                    for rr, cc in region:
                        new_grid.values[rr][cc] = input_grid.values[rr][cc]

    return new_grid
