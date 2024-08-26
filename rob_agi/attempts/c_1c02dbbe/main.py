from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_1c02dbbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions based on seed points.
    
    The function identifies seed points (non-gray, non-black cells) in the order they appear
    from top to bottom, left to right. It then expands each color from its seed point,
    stopping at non-gray colors or black cells. The transformation preserves the original
    structure including black cells and borders.
    
    Steps:
    1. Find all seed points in the grid
    2. Expand colors from each seed point using breadth-first search
    3. Preserve original black cells and borders
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with expanded color regions
    """
    output_grid = input_grid.deep_copy()
    
    seed_points = find_seed_points(output_grid)
    
    for color, row, col in seed_points:
        expand_color(output_grid, color, row, col)
    
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

def expand_color(grid: ColoredGrid, color: int, start_row: int, start_col: int):
    rows, cols = grid.get_dimensions()
    queue = deque([(start_row, start_col)])
    visited = set()

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if grid.get_cell(r, c) == 5:  # Only change if it's gray
            grid.set_cell(r, c, color)

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell_color = grid.get_cell(nr, nc)
                if cell_color == 5:  # Only expand to gray cells
                    queue.append((nr, nc))

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
