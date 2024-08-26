from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_f9a67cb5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal red structure that connects all blue squares.
    
    1. Analyze the grid to identify blue segments and their coordinates.
    2. Determine the optimal vertical backbone position.
    3. Create the vertical backbone connecting all blue segments.
    4. Connect isolated blue segments to the backbone.
    5. Handle the initial red square (if present) by connecting it to the structure.
    6. Optimize the red structure by removing unnecessary red squares.
    7. Validate the final structure to ensure all blue squares are connected.
    
    Returns a new grid with the minimal red structure added while preserving all blue squares.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Find all blue squares
    blue_squares = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 8]
    
    # Determine optimal backbone position
    backbone_pos = min(range(cols), key=lambda c: sum(abs(c - bc) for _, bc in blue_squares))
    
    # Create vertical backbone
    for r in range(rows):
        if any(bc == backbone_pos for _, bc in blue_squares):
            if output_grid.values[r][backbone_pos] != 8:
                output_grid.values[r][backbone_pos] = 2
    
    # Connect blue segments to backbone
    for r, c in blue_squares:
        if c != backbone_pos:
            for cc in range(min(c, backbone_pos), max(c, backbone_pos) + 1):
                if output_grid.values[r][cc] != 8:
                    output_grid.values[r][cc] = 2
    
    # Connect initial red square if present
    initial_red = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 2), None)
    if initial_red:
        r, c = initial_red
        while c != backbone_pos:
            c += 1 if c < backbone_pos else -1
            if output_grid.values[r][c] != 8:
                output_grid.values[r][c] = 2
        while output_grid.values[r][backbone_pos] != 2:
            r += 1
            if output_grid.values[r][backbone_pos] != 8:
                output_grid.values[r][backbone_pos] = 2
    
    # Optimize red structure
    def is_connected(grid: ColoredGrid) -> bool:
        start = next((r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] in {2, 8})
        queue = deque([start])
        visited = set()
        while queue:
            r, c = queue.popleft()
            if (r, c) not in visited:
                visited.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] in {2, 8}:
                        queue.append((nr, nc))
        return all((r, c) in visited for r in range(rows) for c in range(cols) if grid.values[r][c] in {2, 8})
    
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 2:
                temp_grid = output_grid.deep_copy()
                temp_grid.values[r][c] = 0
                if is_connected(temp_grid):
                    output_grid.values[r][c] = 0
    
    return output_grid
