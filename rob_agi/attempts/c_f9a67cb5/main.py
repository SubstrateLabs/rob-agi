from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_f9a67cb5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal red structure that connects all blue squares.
    
    1. Analyze the grid to determine the backbone direction (vertical or horizontal).
    2. Place the backbone along the edge with the most adjacent blue squares.
    3. Connect all blue regions to the backbone using minimal paths.
    4. Optimize the red structure by removing unnecessary red squares.
    5. Ensure the initial red square (if present) is connected to the structure.
    
    Returns a new grid with the minimal red structure added while preserving all blue squares.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    blue_regions = input_grid.find_connected_regions(8)
    
    # Analyze the grid
    blue_cols = set(c for region in blue_regions for _, c in region)
    blue_rows = set(r for region in blue_regions for r, _ in region)
    
    # Determine backbone direction
    backbone_vertical = len(blue_cols) > len(blue_rows)
    
    # Place the backbone
    if backbone_vertical:
        backbone_col = max(range(cols), key=lambda c: sum(c in (0, cols-1) or any(r in blue_rows for r in range(rows) if output_grid.values[r][c-1] == 8 or output_grid.values[r][c+1] == 8)))
        for r in range(rows):
            if output_grid.values[r][backbone_col] != 8:
                output_grid.values[r][backbone_col] = 2
    else:
        backbone_row = max(range(rows), key=lambda r: sum(r in (0, rows-1) or any(c in blue_cols for c in range(cols) if output_grid.values[r-1][c] == 8 or output_grid.values[r+1][c] == 8)))
        for c in range(cols):
            if output_grid.values[backbone_row][c] != 8:
                output_grid.values[backbone_row][c] = 2
    
    # Connect blue regions to the backbone
    for region in blue_regions:
        if not any(output_grid.values[r][c] == 2 for r, c in region):
            r, c = min(region, key=lambda pos: abs(pos[0] - backbone_row) if not backbone_vertical else abs(pos[1] - backbone_col))
            if backbone_vertical:
                for col in range(min(c, backbone_col), max(c, backbone_col) + 1):
                    if output_grid.values[r][col] != 8:
                        output_grid.values[r][col] = 2
            else:
                for row in range(min(r, backbone_row), max(r, backbone_row) + 1):
                    if output_grid.values[row][c] != 8:
                        output_grid.values[row][c] = 2
    
    # Optimize the red structure
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 2:
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                red_neighbors = sum(1 for nr, nc in neighbors if 0 <= nr < rows and 0 <= nc < cols and output_grid.values[nr][nc] == 2)
                if red_neighbors <= 1:
                    output_grid.values[r][c] = 0
    
    # Connect initial red square if present
    initial_red = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 2), None)
    if initial_red:
        r, c = initial_red
        target = min((nr, nc) for nr in range(rows) for nc in range(cols) if output_grid.values[nr][nc] == 2, key=lambda pos: abs(pos[0] - r) + abs(pos[1] - c))
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
    
    return output_grid
