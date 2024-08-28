from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_1c02dbbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions based on seed points.
    
    The function identifies seed points (non-gray, non-black cells) and expands each color
    simultaneously using a distance-based approach. Colors expand equally in all directions,
    stopping at non-gray colors, black cells, or when meeting other expanding colors.
    The transformation preserves the original structure including black cells and borders.
    
    Steps:
    1. Identify all seed points in the grid
    2. Expand colors simultaneously using a distance-based breadth-first search
    3. Preserve original black cells and borders
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with expanded color regions
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    seed_points = find_seed_points(output_grid)
    queue = deque([(r, c, color, 0) for color, r, c in seed_points])
    distances = {(r, c): float('inf') for r in range(rows) for c in range(cols)}
    
    while queue:
        r, c, color, dist = queue.popleft()
        if output_grid.get_cell(r, c) not in [0, 5] and (r, c) not in [(sr, sc) for _, sr, sc in seed_points]:
            continue
        if distances[(r, c)] <= dist:
            continue
        
        distances[(r, c)] = dist
        output_grid.set_cell(r, c, color)
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and output_grid.get_cell(nr, nc) == 5:
                queue.append((nr, nc, color, dist + 1))
    
    preserve_original_structure(output_grid, input_grid)
    return output_grid

def find_seed_points(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    seed_points = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color not in [0, 5]:  # Not black or gray
                seed_points.append((color, r, c))
    return seed_points

def preserve_original_structure(output_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = output_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) == 0:  # If originally black
                output_grid.set_cell(r, c, 0)  # Keep it black
    # Ensure border is black
    for r in range(rows):
        output_grid.set_cell(r, 0, 0)
        output_grid.set_cell(r, cols-1, 0)
    for c in range(cols):
        output_grid.set_cell(0, c, 0)
        output_grid.set_cell(rows-1, c, 0)
