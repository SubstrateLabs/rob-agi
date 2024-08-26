from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f8be4b64(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored crosses into territories.
    
    1. Identifies colored crosses in the input grid.
    2. Determines color priority (descending order, with green always last).
    3. Calculates territories for each cross.
    4. Fills a new grid based on territories and color priority.
    5. Restores original green crosses.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    crosses = find_crosses(input_grid)
    color_priority = get_color_priority(crosses)
    territories = calculate_territories(crosses, rows, cols)
    
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    for color in color_priority:
        for cross in crosses:
            if cross[2] == color:
                fill_territory(new_grid, territories[cross[:2]], color)
    
    restore_green_crosses(new_grid, crosses)
    
    return new_grid

def find_crosses(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    crosses = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                if is_cross_center(grid, r, c):
                    crosses.append((r, c, grid.values[r][c]))
    return crosses

def is_cross_center(grid: ColoredGrid, r: int, c: int) -> bool:
    color = grid.values[r][c]
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 3:
            return True
    return False

def get_color_priority(crosses: List[Tuple[int, int, int]]) -> List[int]:
    colors = set(cross[2] for cross in crosses)
    priority = sorted([c for c in colors if c != 3], reverse=True)
    if 3 in colors:
        priority.append(3)
    return priority

def calculate_territories(crosses: List[Tuple[int, int, int]], rows: int, cols: int) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    territories = {}
    for r, c, _ in crosses:
        top = bottom = r
        left = right = c
        for other_r, other_c, _ in crosses:
            if other_r < r and other_r > top:
                top = (r + other_r) // 2
            elif other_r > r and other_r < bottom:
                bottom = (r + other_r) // 2
            if other_c < c and other_c > left:
                left = (c + other_c) // 2
            elif other_c > c and other_c < right:
                right = (c + other_c) // 2
        top = max(0, top)
        left = max(0, left)
        bottom = min(rows - 1, bottom)
        right = min(cols - 1, right)
        territories[(r, c)] = (top, left, bottom, right)
    return territories

def fill_territory(grid: ColoredGrid, territory: Tuple[int, int, int, int], color: int):
    top, left, bottom, right = territory
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            grid.values[r][c] = color

def restore_green_crosses(grid: ColoredGrid, crosses: List[Tuple[int, int, int]]):
    for r, c, color in crosses:
        if color == 3:
            grid.values[r][c] = 3
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]):
                    grid.values[nr][nc] = 3
