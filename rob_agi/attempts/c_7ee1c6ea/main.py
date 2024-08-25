from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7ee1c6ea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by swapping the two most frequent colors within each region
    separated by gray (5) structures. Black (0) and gray (5) squares remain unchanged.
    
    1. Identifies regions separated by gray structures.
    2. For each region, determines the two most frequent colors.
    3. Swaps these colors within the region.
    4. Preserves black and gray squares.
    
    Returns the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    regions = find_regions(new_grid)
    
    for region in regions:
        color1, color2 = get_swap_colors(region, input_grid)
        swap_map = {color1: color2, color2: color1}
        
        for row, col in region:
            current_color = new_grid.values[row][col]
            if current_color in [color1, color2]:
                new_grid.values[row][col] = swap_map[current_color]
    
    return new_grid

def find_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    regions = []
    visited = set()
    rows, cols = grid.get_dimensions()

    def dfs(r, c, region):
        if (r, c) in visited or grid.values[r][c] == 5:
            return
        visited.add((r, c))
        region.append((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                dfs(nr, nc, region)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 5:
                region = []
                dfs(r, c, region)
                regions.append(region)

    return regions

def get_swap_colors(region: List[Tuple[int, int]], grid: ColoredGrid) -> Tuple[int, int]:
    color_count = {}
    for r, c in region:
        color = grid.values[r][c]
        if color not in [0, 5]:
            color_count[color] = color_count.get(color, 0) + 1
    sorted_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_colors[0][0], sorted_colors[1][0] if len(sorted_colors) > 1 else (sorted_colors[0][0], -1)
