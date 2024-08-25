from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_c6e1b8da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adjusting the position and shape of colored regions.
    
    The transformation follows these steps:
    1. Identify colored regions in the input grid.
    2. Determine bounding rectangles for each region.
    3. Compress regions into rectangles.
    4. Optimize layout by placing regions compactly.
    5. Ensure a border of empty space around the edges.
    6. Fine-tune placement to improve alignment and spacing.
    7. Reconstruct the output grid with transformed regions.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    regions = identify_regions(input_grid)
    compressed_regions = compress_regions(regions)
    layout = optimize_layout(compressed_regions, input_grid.get_dimensions())
    output_grid = reconstruct_grid(layout, input_grid.get_dimensions())
    return ensure_border(output_grid)

def identify_regions(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    regions = {}
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                color = grid.get_cell(r, c)
                region = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                                stack.append((nr, nc))
                if color not in regions:
                    regions[color] = []
                regions[color].append(region)
    return regions

def compress_regions(regions: Dict[int, List[List[Tuple[int, int]]]]) -> Dict[int, List[Tuple[int, int, int, int]]]:
    compressed = {}
    for color, color_regions in regions.items():
        compressed[color] = []
        for region in color_regions:
            min_r = min(r for r, _ in region)
            max_r = max(r for r, _ in region)
            min_c = min(c for _, c in region)
            max_c = max(c for _, c in region)
            compressed[color].append((min_r, min_c, max_r - min_r + 1, max_c - min_c + 1))
    return compressed

def optimize_layout(regions: Dict[int, List[Tuple[int, int, int, int]]], dimensions: Tuple[int, int]) -> List[Tuple[int, int, int, int, int]]:
    rows, cols = dimensions
    layout = []
    for color, rectangles in regions.items():
        for r, c, h, w in rectangles:
            layout.append((color, r, c, h, w))
    
    layout.sort(key=lambda x: x[3] * x[4], reverse=True)  # Sort by area
    
    optimized = []
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for color, _, _, h, w in layout:
        placed = False
        for r in range(rows - h + 1):
            if placed:
                break
            for c in range(cols - w + 1):
                if all(grid[rr][cc] == 0 for rr in range(r, r+h) for cc in range(c, c+w)):
                    optimized.append((color, r, c, h, w))
                    for rr in range(r, r+h):
                        for cc in range(c, c+w):
                            grid[rr][cc] = color
                    placed = True
                    break
        if not placed:
            # If we can't place the region, we'll skip it
            pass
    
    return optimized

def reconstruct_grid(layout: List[Tuple[int, int, int, int, int]], dimensions: Tuple[int, int]) -> ColoredGrid:
    rows, cols = dimensions
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for color, r, c, h, w in layout:
        for rr in range(r, r+h):
            for cc in range(c, c+w):
                if 0 <= rr < rows and 0 <= cc < cols:
                    grid[rr][cc] = color
    return ColoredGrid(values=grid)

def ensure_border(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            new_grid[r][c] = grid.get_cell(r, c)
    return ColoredGrid(values=new_grid)
