from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_67c52801(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving colored cell groups downward and to the left.
    
    The transformation follows these rules:
    1. The bottom row of the input grid remains unchanged.
    2. Colored cell groups move downward and to the left, maintaining their shapes.
    3. Groups are placed in order from bottom to top, then left to right.
    4. Single colored cells in the second-to-last row move as far left as possible.
    5. Empty space (black/0) fills from the top and right.
    
    The algorithm works as follows:
    1. Initialize the output grid with zeros and copy the bottom row from the input.
    2. Identify and group connected colored cells.
    3. Sort groups based on their bottom position and leftmost cell.
    4. Place each group in the lowest, leftmost available position that fits the entire group.
    5. Handle single cells in the second-to-last row.
    6. Return the transformed grid.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified rules.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Copy the bottom row
    output_grid.values[-1] = input_grid.values[-1].copy()
    
    # Identify color groups
    color_groups = identify_color_groups(input_grid)
    
    # Sort color groups
    color_groups.sort(key=lambda g: (-max(r for r, _, _ in g), min(c for _, c, _ in g)))
    
    # Place color groups
    for group in color_groups:
        place_group(output_grid, group)
    
    # Handle single cells in second-to-last row
    handle_single_cells(input_grid, output_grid)
    
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
    min_col = min(c for _, c, _ in group)
    
    for placement_row in range(rows - 2, -1, -1):  # Start from second to last row
        for placement_col in range(min_col, cols):  # Start from the original leftmost position
            if can_place_group(output_grid, group, placement_row, placement_col, group_height, group_width):
                for r, c, color in group:
                    rel_r, rel_c = r - min(r for r, _, _ in group), c - min(c for _, c, _ in group)
                    output_grid.values[placement_row + rel_r][placement_col + rel_c] = color
                return

def can_place_group(grid: ColoredGrid, group: List[Tuple[int, int, int]], start_row: int, start_col: int, height: int, width: int) -> bool:
    rows, cols = grid.get_dimensions()
    if start_row + height > rows - 1 or start_col + width > cols:
        return False
    
    for r, c, _ in group:
        rel_r, rel_c = r - min(r for r, _, _ in group), c - min(c for _, c, _ in group)
        if grid.values[start_row + rel_r][start_col + rel_c] != 0:
            return False
    
    return True

def handle_single_cells(input_grid: ColoredGrid, output_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    second_last_row = rows - 2
    
    for c in range(cols):
        if input_grid.values[second_last_row][c] != 0 and output_grid.values[second_last_row][c] == 0:
            color = input_grid.values[second_last_row][c]
            for new_c in range(c, cols):
                if output_grid.values[second_last_row][new_c] == 0:
                    output_grid.values[second_last_row][new_c] = color
                    break
