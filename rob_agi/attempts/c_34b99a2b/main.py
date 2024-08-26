from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_34b99a2b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by analyzing patterns of sky (8) and gray (5) regions
    separated by yellow (4) columns, and creates a simplified 5x4 output grid.
    
    The algorithm works as follows:
    1. Split the input grid into left and right halves using the yellow columns.
    2. Detect connected regions of sky (8) in the left half and gray (5) in the right half.
    3. Analyze the significance of these regions based on size and position.
    4. Map significant regions to a 5x4 output grid, representing them with red (2) cells.
    5. Apply final adjustments to match the observed patterns in the examples.
    """
    rows, cols = input_grid.get_dimensions()
    left_half, right_half = split_grid(input_grid)
    
    left_regions = find_connected_regions(left_half, 8)
    right_regions = find_connected_regions(right_half, 5)
    
    output = [[0 for _ in range(4)] for _ in range(5)]
    
    map_regions_to_output(left_regions, output, 0, 2)
    map_regions_to_output(right_regions, output, 2, 4)
    
    apply_final_adjustments(output)
    
    return ColoredGrid(values=output)

def split_grid(grid: ColoredGrid) -> Tuple[List[List[int]], List[List[int]]]:
    rows, cols = grid.get_dimensions()
    mid = cols // 2
    left_half = [row[:mid] for row in grid.values]
    right_half = [row[mid+1:] for row in grid.values]
    return left_half, right_half

def find_connected_regions(grid: List[List[int]], color: int) -> List[List[Tuple[int, int]]]:
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] == color:
                region = []
                dfs(grid, r, c, color, visited, region)
                if region:
                    regions.append(region)
    
    return regions

def dfs(grid: List[List[int]], r: int, c: int, color: int, visited: List[List[bool]], region: List[Tuple[int, int]]):
    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or visited[r][c] or grid[r][c] != color:
        return
    
    visited[r][c] = True
    region.append((r, c))
    
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, color, visited, region)

def map_regions_to_output(regions: List[List[Tuple[int, int]]], output: List[List[int]], start_col: int, end_col: int):
    for region in regions:
        if len(region) >= 3:  # Consider regions with at least 3 cells as significant
            for r, c in region:
                if start_col <= c < end_col:
                    output_col = start_col + (c - start_col) // 2
                    output[r][output_col] = 2

def apply_final_adjustments(output: List[List[int]]):
    # Fill columns with 2 or more red cells
    for c in range(4):
        if sum(row[c] for row in output) >= 2:
            for r in range(5):
                output[r][c] = 2
    
    # Remove isolated red cells
    for r in range(5):
        for c in range(4):
            if output[r][c] == 2:
                if sum(output[r][max(0, c-1):min(4, c+2)]) == 2:
                    output[r][c] = 0
