from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_a57f2f04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying specific patterns to distinct regions.
    
    The function identifies non-sky blue regions in the input grid and replaces them
    with specific patterns based on the color of non-black elements:
    - For red (2): 3x3 pattern [[2,0,2], [2,2,2], [0,2,2]] repeating vertically and horizontally.
    - For green (3): 3x3 pattern with color in corners and center [[3,0,3], [0,3,0], [3,0,3]].
    - For other colors: 2x2 checkerboard pattern, starting with the color.
    The sky blue (8) background remains unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed output grid.
    """
    output_grid = input_grid.deep_copy()
    regions = find_regions(input_grid)
    
    for region in regions:
        color = determine_color(input_grid, region)
        if color is not None:
            pattern = generate_pattern(color)
            apply_pattern(output_grid, pattern, region)
    
    return output_grid

def find_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    regions = []
    rows, cols = grid.get_dimensions()
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    def dfs(r, c):
        region = []
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < rows and 0 <= c < cols and not visited[r][c] and grid.values[r][c] != 8:
                visited[r][c] = True
                region.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        return region
    
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid.values[i][j] != 8:
                regions.append(dfs(i, j))
    
    return regions

def determine_color(grid: ColoredGrid, region: List[Tuple[int, int]]) -> Optional[int]:
    for r, c in region:
        color = grid.values[r][c]
        if color not in {0, 8}:
            return color
    return None

def generate_pattern(color: int) -> List[List[int]]:
    if color == 2:  # Red
        return [[2, 0, 2], [2, 2, 2], [0, 2, 2]]
    elif color == 3:  # Green
        return [[3, 0, 3], [0, 3, 0], [3, 0, 3]]
    else:
        return [[color, 0], [0, color]]

def apply_pattern(grid: ColoredGrid, pattern: List[List[int]], region: List[Tuple[int, int]]):
    pattern_height, pattern_width = len(pattern), len(pattern[0])
    min_r = min(r for r, _ in region)
    min_c = min(c for _, c in region)
    
    for r, c in region:
        pr = (r - min_r) % pattern_height
        pc = (c - min_c) % pattern_width
        grid.values[r][c] = pattern[pr][pc]
