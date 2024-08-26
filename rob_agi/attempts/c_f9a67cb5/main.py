from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_f9a67cb5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal red structure that connects all blue squares.
    
    1. Analyze the grid to identify blue regions and potential backbone placements.
    2. Evaluate multiple backbone positions to find the most efficient placement.
    3. Create the red structure by placing the backbone and connecting all blue regions.
    4. Handle the initial red square (if present) by connecting it to the structure.
    5. Optimize the red structure by removing unnecessary red squares.
    6. Validate the final structure to ensure all blue squares are connected.
    
    Returns a new grid with the minimal red structure added while preserving all blue squares.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    blue_regions = input_grid.find_connected_regions(8)
    
    def evaluate_backbone(is_vertical: bool, position: int) -> int:
        red_count = 0
        for r in range(rows):
            for c in range(cols):
                if (is_vertical and c == position) or (not is_vertical and r == position):
                    if output_grid.values[r][c] != 8:
                        red_count += 1
        
        for region in blue_regions:
            if not any((is_vertical and c == position) or (not is_vertical and r == position) for r, c in region):
                red_count += min(abs(position - (c if is_vertical else r)) for r, c in region)
        
        return red_count
    
    # Evaluate backbone placements
    vertical_backbones = [(True, c, evaluate_backbone(True, c)) for c in range(cols)]
    horizontal_backbones = [(False, r, evaluate_backbone(False, r)) for r in range(rows)]
    is_vertical, backbone_pos, _ = min(vertical_backbones + horizontal_backbones, key=lambda x: x[2])
    
    # Place the backbone
    for r in range(rows):
        for c in range(cols):
            if (is_vertical and c == backbone_pos) or (not is_vertical and r == backbone_pos):
                if output_grid.values[r][c] != 8:
                    output_grid.values[r][c] = 2
    
    # Connect blue regions to the backbone
    for region in blue_regions:
        if not any(output_grid.values[r][c] == 2 for r, c in region):
            target = min(region, key=lambda pos: abs(pos[1 if is_vertical else 0] - backbone_pos))
            r, c = target
            while (is_vertical and c != backbone_pos) or (not is_vertical and r != backbone_pos):
                if is_vertical:
                    c += 1 if c < backbone_pos else -1
                else:
                    r += 1 if r < backbone_pos else -1
                if output_grid.values[r][c] != 8:
                    output_grid.values[r][c] = 2
    
    # Connect initial red square if present
    initial_red = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 2), None)
    if initial_red:
        r, c = initial_red
        target = min(((nr, nc) for nr in range(rows) for nc in range(cols) if output_grid.values[nr][nc] == 2), 
                     key=lambda pos: abs(pos[0] - r) + abs(pos[1] - c))
        while (r, c) != target:
            if r < target[0]:
                r += 1
            elif r > target[0]:
                r -= 1
            elif c < target[1]:
                c += 1
            elif c > target[1]:
                c -= 1
            if output_grid.values[r][c] != 8:
                output_grid.values[r][c] = 2
    
    # Optimize the red structure
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
