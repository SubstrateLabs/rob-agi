from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def find_connected_regions(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    regions = []

    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            visited[r][c] or grid.values[r][c] != color):
            return []
        
        visited[r][c] = True
        region = [(r, c)]
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
            nr, nc = r + dr, c + dc
            region.extend(dfs(nr, nc))
        
        return region

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid.values[r][c] == color:
                regions.append(dfs(r, c))

    return regions

def solve_ae58858e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing red (2) regions of more than 3 connected squares to magenta (6).
    Red regions with 3 or fewer squares remain unchanged. Connected regions are defined as orthogonally adjacent squares.
    """
    # Create a deep copy of the input grid
    grid = input_grid.deep_copy()
    
    # Find all red regions
    red_regions = find_connected_regions(grid, color=2)
    
    # Process each region
    for region in red_regions:
        if len(region) > 3:
            # Change to magenta if region has more than 3 squares
            for r, c in region:
                grid.values[r][c] = 6
    
    return grid
