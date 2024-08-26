from rob_agi.colored_grid import ColoredGrid
from typing import List, Set, Tuple, Dict
from collections import deque

def solve_7d1f7ee8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the interiors of color regions with their respective colors.
    
    The function works as follows:
    1. Identifies all unique colors and their boundary cells.
    2. Sorts colors based on their "outerness" (proximity to grid edges).
    3. For each color, fills its interior regions using flood fill.
    4. Preserves nested shapes and handles partial frames.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def get_boundary_cells(color: int) -> Set[Tuple[int, int]]:
        boundary = set()
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == color:
                    if is_border(r, c) or any(output_grid.values[nr][nc] != color 
                                              for nr, nc in get_neighbors(r, c)):
                        boundary.add((r, c))
        return boundary
    
    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1
    
    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]
    
    def flood_fill(start: Tuple[int, int], color: int):
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = color
                queue.extend(get_neighbors(r, c))
    
    # Identify unique colors and their boundary cells
    colors = set(output_grid.values[r][c] for r in range(rows) for c in range(cols)) - {0}
    color_boundaries = {color: get_boundary_cells(color) for color in colors}
    
    # Sort colors based on "outerness"
    sorted_colors = sorted(colors, key=lambda c: min(r + c for r, c in color_boundaries[c]))
    
    # Process each color
    for color in sorted_colors:
        for r, c in color_boundaries[color]:
            for nr, nc in get_neighbors(r, c):
                if output_grid.values[nr][nc] == 0:
                    flood_fill((nr, nc), color)
    
    return output_grid
