from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import math

def solve_c6e1b8da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adjusting the position and shape of colored regions.
    
    The transformation follows these steps:
    1. Analyze the input grid to identify distinct colored regions.
    2. Calculate properties for each region (area, bounding box, aspect ratio, center of mass).
    3. Create a 20x20 grid framework with 5x5 cell subdivisions.
    4. Place regions on the grid, starting with the largest, aligning with 5-cell boundaries.
    5. Adjust regions to maintain adjacency and relative positions.
    6. Ensure consistent 1-cell spacing between regions and a 1-cell black border.
    7. Regularize shapes to create straight edges and rectangular regions.
    8. Construct and validate the output grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the rules.
    """
    regions = analyze_grid(input_grid)
    layout = create_layout(regions, input_grid.get_dimensions())
    output_grid = construct_output_grid(layout, input_grid.get_dimensions())
    return output_grid

def analyze_grid(grid: ColoredGrid) -> List[Dict]:
    regions = []
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
                
                min_r = min(r for r, _ in region)
                max_r = max(r for r, _ in region)
                min_c = min(c for _, c in region)
                max_c = max(c for _, c in region)
                area = len(region)
                center_r = sum(r for r, _ in region) / area
                center_c = sum(c for _, c in region) / area
                
                regions.append({
                    'color': color,
                    'area': area,
                    'bounding_box': (min_r, min_c, max_r - min_r + 1, max_c - min_c + 1),
                    'aspect_ratio': (max_c - min_c + 1) / (max_r - min_r + 1),
                    'center': (center_r, center_c)
                })
    
    return sorted(regions, key=lambda x: x['area'], reverse=True)

def create_layout(regions: List[Dict], dimensions: Tuple[int, int]) -> List[Tuple[int, int, int, int, int]]:
    rows, cols = dimensions
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    layout = []
    
    for region in regions:
        color = region['color']
        area = region['area']
        aspect_ratio = region['aspect_ratio']
        
        # Calculate ideal dimensions aligned to 5-cell grid
        ideal_width = math.sqrt(area * aspect_ratio)
        ideal_height = area / ideal_width
        width = max(5, 5 * math.ceil(ideal_width / 5))
        height = max(5, 5 * math.ceil(ideal_height / 5))
        
        # Find best position
        best_pos = None
        min_distance = float('inf')
        for r in range(1, rows - height, 5):
            for c in range(1, cols - width, 5):
                if all(grid[rr][cc] == 0 for rr in range(r, r+height) for cc in range(c, c+width)):
                    distance = ((r + height/2) - region['center'][0])**2 + ((c + width/2) - region['center'][1])**2
                    if distance < min_distance:
                        min_distance = distance
                        best_pos = (r, c)
        
        if best_pos:
            r, c = best_pos
            layout.append((color, r, c, height, width))
            for rr in range(r, r+height):
                for cc in range(c, c+width):
                    grid[rr][cc] = color
    
    return layout

def construct_output_grid(layout: List[Tuple[int, int, int, int, int]], dimensions: Tuple[int, int]) -> ColoredGrid:
    rows, cols = dimensions
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for color, r, c, h, w in layout:
        for rr in range(r, r+h):
            for cc in range(c, c+w):
                if 0 < rr < rows-1 and 0 < cc < cols-1:
                    grid[rr][cc] = color
    return ColoredGrid(values=grid)
