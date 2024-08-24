from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_4347f46a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by hollowing out colored regions while preserving their borders.
    
    This function identifies connected regions of the same color, determines their bounding boxes,
    and sets interior cells to 0 (black) while preserving the border. Small regions (1x1, 2x2) are
    not modified. The transformation is applied to all colored regions in the grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with hollowed out regions.
    """
    def flood_fill(grid: List[List[int]], row: int, col: int, color: int) -> List[Tuple[int, int]]:
        height, width = len(grid), len(grid[0])
        queue = deque([(row, col)])
        visited = set([(row, col)])
        region = []
        
        while queue:
            r, c = queue.popleft()
            region.append((r, c))
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] == color and (nr, nc) not in visited:
                    queue.append((nr, nc))
                    visited.add((nr, nc))
        
        return region

    def hollow_out_region(grid: List[List[int]], region: List[Tuple[int, int]]) -> None:
        if len(region) <= 4:  # Don't modify small regions
            return
        
        min_row = min(r for r, _ in region)
        max_row = max(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_col = max(c for _, c in region)
        
        for r in range(min_row + 1, max_row):
            for c in range(min_col + 1, max_col):
                if (r, c) in region:
                    # Check if it's not on the border
                    if (r-1, c) in region and (r+1, c) in region and (r, c-1) in region and (r, c+1) in region:
                        grid[r][c] = 0

    # Create a deep copy of the input grid
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()
    
    # Process each cell in the grid
    for row in range(height):
        for col in range(width):
            color = grid.get_cell(row, col)
            if color != 0:
                region = flood_fill(grid.values, row, col, color)
                hollow_out_region(grid.values, region)
    
    return grid
