from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_5b692c0f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating symmetry in shapes and expanding them.
    
    The function performs the following steps:
    1. Identifies connected regions (shapes) in the input grid.
    2. For each shape:
       a. Determines its bounding box and primary axis (horizontal or vertical).
       b. Creates symmetry by mirroring along the primary axis.
       c. Expands the shape to fill its bounding box.
       d. Smooths the edges to create more cohesive shapes.
    3. Places the transformed shapes onto a new grid.
    
    This results in more symmetrical and expanded versions of the original shapes,
    while maintaining their relative positions and color patterns.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    regions = find_connected_regions(input_grid)
    for region in regions:
        transformed_shape = transform_shape(region)
        place_shape(output_grid, transformed_shape)
    
    return output_grid

def find_connected_regions(grid: ColoredGrid) -> List[List[Tuple[int, int, int]]]:
    regions = []
    visited = set()
    
    def flood_fill(r: int, c: int, color: int) -> List[Tuple[int, int, int]]:
        region = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and 0 <= curr_r < grid.num_rows and 0 <= curr_c < grid.num_cols and grid.get_cell(curr_r, curr_c) == color:
                visited.add((curr_r, curr_c))
                region.append((curr_r, curr_c, color))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((curr_r + dr, curr_c + dc))
        return region

    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                regions.append(flood_fill(r, c, grid.get_cell(r, c)))
    
    return regions

def transform_shape(region: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    if not region:
        return []
    
    # Determine bounding box
    top = min(r for r, _, _ in region)
    bottom = max(r for r, _, _ in region)
    left = min(c for _, c, _ in region)
    right = max(c for _, c, _ in region)
    
    width = right - left + 1
    height = bottom - top + 1
    
    # Determine primary axis
    if width > height:
        symmetrical_shape = mirror_horizontal(region, top, bottom)
    else:
        symmetrical_shape = mirror_vertical(region, left, right)
    
    # Expand shape
    expanded_shape = expand_shape(symmetrical_shape, top, left, bottom, right)
    
    # Smooth edges
    smoothed_shape = smooth_edges(expanded_shape, top, left, bottom, right)
    
    return smoothed_shape

def mirror_horizontal(shape: List[Tuple[int, int, int]], top: int, bottom: int) -> List[Tuple[int, int, int]]:
    midline = (top + bottom) // 2
    mirrored = shape.copy()
    for r, c, color in shape:
        if r <= midline:
            mirrored.append((2 * midline - r, c, color))
    return mirrored

def mirror_vertical(shape: List[Tuple[int, int, int]], left: int, right: int) -> List[Tuple[int, int, int]]:
    midline = (left + right) // 2
    mirrored = shape.copy()
    for r, c, color in shape:
        if c <= midline:
            mirrored.append((r, 2 * midline - c, color))
    return mirrored

def expand_shape(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    expanded = shape.copy()
    shape_set = set((r, c) for r, c, _ in shape)
    
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c) not in shape_set:
                neighbors = [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (r+dr, c+dc) in shape_set]
                if neighbors:
                    color_counts = defaultdict(int)
                    for nr, nc in neighbors:
                        color = next(color for x, y, color in shape if x == nr and y == nc)
                        color_counts[color] += 1
                    most_common_color = max(color_counts, key=color_counts.get)
                    expanded.append((r, c, most_common_color))
    
    return expanded

def smooth_edges(shape: List[Tuple[int, int, int]], top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int]]:
    smoothed = shape.copy()
    shape_dict = {(r, c): color for r, c, color in shape}
    
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c) not in shape_dict:
                neighbors = [(r+dr, c+dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (r+dr, c+dc) in shape_dict]
                if len(neighbors) >= 5:
                    color_counts = defaultdict(int)
                    for nr, nc in neighbors:
                        color_counts[shape_dict[(nr, nc)]] += 1
                    most_common_color = max(color_counts, key=color_counts.get)
                    smoothed.append((r, c, most_common_color))
    
    return smoothed

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int, int]]):
    for r, c, color in shape:
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
            grid.set_cell(r, c, color)
