from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_67c52801(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving colored cell groups downward and to the left.
    
    The transformation follows these rules:
    1. The bottom row of the input grid remains unchanged.
    2. Colored cell groups move downward and to the left, maintaining their shapes and relative order.
    3. Empty space (black/0) fills from the top and right.
    
    The algorithm works as follows:
    1. Initialize the output grid with zeros and copy the bottom row from the input.
    2. Identify and group connected colored cells.
    3. Place each group in the lowest, leftmost available position that fits the entire group.
    4. Preserve the relative order of groups.
    5. Return the transformed grid.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified rules.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy the bottom row
    output_grid.values[-1] = input_grid.values[-1].copy()
    
    # Identify color groups
    color_groups = identify_color_groups(input_grid)
    
    # Place color groups
    for group in color_groups:
        place_group(output_grid, group)
    
    return output_grid

def identify_color_groups(grid: ColoredGrid) -> List[List[Tuple[int, int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    groups = []
    
    for r in range(rows - 1):  # Exclude bottom row
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                group = []
                dfs(grid, r, c, grid.values[r][c], visited, group)
                groups.append(group)
    
    return groups

def dfs(grid: ColoredGrid, r: int, c: int, color: int, visited: set, group: List[Tuple[int, int, int]]):
    if r < 0 or r >= grid.num_rows - 1 or c < 0 or c >= grid.num_cols or (r, c) in visited or grid.values[r][c] != color:
        return
    
    visited.add((r, c))
    group.append((r, c, color))
    
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, color, visited, group)

def place_group(output_grid: ColoredGrid, group: List[Tuple[int, int, int]]):
    rows, cols = output_grid.get_dimensions()
    group_height = max(r for r, _, _ in group) - min(r for r, _, _ in group) + 1
    group_width = max(c for _, c, _ in group) - min(c for _, c, _ in group) + 1
    
    for placement_row in range(rows - 2, -1, -1):  # Start from second to last row
        for placement_col in range(cols - group_width + 1):
            if can_place_group(output_grid, group, placement_row, placement_col, group_height, group_width):
                for r, c, color in group:
                    rel_r, rel_c = r - min(r for r, _, _ in group), c - min(c for _, c, _ in group)
                    output_grid.values[placement_row + rel_r][placement_col + rel_c] = color
                return

def can_place_group(grid: ColoredGrid, group: List[Tuple[int, int, int]], start_row: int, start_col: int, height: int, width: int) -> bool:
    if start_row + height > grid.num_rows - 1 or start_col + width > grid.num_cols:
        return False
    
    for r in range(start_row, start_row + height):
        for c in range(start_col, start_col + width):
            if grid.values[r][c] != 0:
                return False
    
    return True
