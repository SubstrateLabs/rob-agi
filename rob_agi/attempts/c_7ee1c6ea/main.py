from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional, Set

def solve_7ee1c6ea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing the color of connected regions to their most frequent adjacent color.
    
    1. Identifies connected regions of the same color.
    2. For each region, determines the most frequent adjacent color (excluding gray and black).
    3. Changes the color of the entire region to the most frequent adjacent color.
    4. Preserves black (0) and gray (5) squares.
    
    Returns the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    processed = set()
    return process_grid(new_grid, processed)

def process_grid(grid: ColoredGrid, processed: Set[Tuple[int, int]]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in processed or grid.values[r][c] in [0, 5]:
                continue
            region = find_connected_region(grid, r, c, grid.values[r][c])
            processed.update(region)
            if len(region) > 1:
                swap_color = find_most_frequent_adjacent_color(grid, region)
                if swap_color is not None:
                    apply_color_swap(grid, region, swap_color)
    return grid

def find_connected_region(grid: ColoredGrid, r: int, c: int, color: int) -> Set[Tuple[int, int]]:
    region = set()
    stack = [(r, c)]
    rows, cols = grid.get_dimensions()
    while stack:
        curr_r, curr_c = stack.pop()
        if (curr_r, curr_c) in region:
            continue
        if 0 <= curr_r < rows and 0 <= curr_c < cols and grid.values[curr_r][curr_c] == color:
            region.add((curr_r, curr_c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((curr_r + dr, curr_c + dc))
    return region

def find_most_frequent_adjacent_color(grid: ColoredGrid, region: Set[Tuple[int, int]]) -> Optional[int]:
    color_freq = {}
    region_color = grid.values[list(region)[0][0]][list(region)[0][1]]
    rows, cols = grid.get_dimensions()
    for r, c in region:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region and 0 <= nr < rows and 0 <= nc < cols:
                adj_color = grid.values[nr][nc]
                if adj_color not in [0, 5, region_color]:
                    color_freq[adj_color] = color_freq.get(adj_color, 0) + 1
    return max(color_freq, key=color_freq.get) if color_freq else None

def apply_color_swap(grid: ColoredGrid, region: Set[Tuple[int, int]], new_color: int):
    for r, c in region:
        grid.values[r][c] = new_color
