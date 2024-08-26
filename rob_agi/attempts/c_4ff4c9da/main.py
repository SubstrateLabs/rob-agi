from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) formations symmetrically.
    
    The function identifies existing sky blue formations and expands them vertically
    and horizontally while maintaining symmetry across both axes. It only replaces
    black (0) or blue (1) cells, preserving the underlying red (2) cell pattern.
    
    Steps:
    1. Identify initial sky blue formations
    2. Expand formations vertically and horizontally
    3. Ensure symmetry across both axes
    4. Fill corner regions if surrounded by sky blue cells
    5. Repeat expansion until no new cells are added
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    center_r, center_c = rows // 2, cols // 2
    
    while True:
        new_cells = expand_formations(grid, center_r, center_c)
        if not new_cells:
            break
    
    fill_corners(grid)
    ensure_symmetry(grid, center_r, center_c)
    
    return grid

def expand_formations(grid: ColoredGrid, center_r: int, center_c: int) -> Set[Tuple[int, int]]:
    new_cells = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                new_cells.update(expand_cell(grid, r, c, rows, cols))
    
    for r, c in new_cells:
        grid.set_cell(r, c, 8)
    
    return new_cells

def expand_cell(grid: ColoredGrid, r: int, c: int, rows: int, cols: int) -> Set[Tuple[int, int]]:
    new_cells = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) in [0, 1]:
            new_cells.add((nr, nc))
    
    return new_cells

def fill_corners(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    
    for r, c in corners:
        if is_corner_surrounded(grid, r, c):
            flood_fill_corner(grid, r, c)

def is_corner_surrounded(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    directions = [(0, 1), (1, 0)] if r == 0 and c == 0 else \
                 [(0, -1), (1, 0)] if r == 0 and c == cols-1 else \
                 [(0, 1), (-1, 0)] if r == rows-1 and c == 0 else \
                 [(0, -1), (-1, 0)]
    
    return all(grid.get_cell(r + dr, c + dc) == 8 for dr, dc in directions)

def flood_fill_corner(grid: ColoredGrid, r: int, c: int):
    rows, cols = grid.get_dimensions()
    stack = [(r, c)]
    
    while stack:
        r, c = stack.pop()
        if grid.get_cell(r, c) in [0, 1]:
            grid.set_cell(r, c, 8)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) in [0, 1]:
                    stack.append((nr, nc))

def ensure_symmetry(grid: ColoredGrid, center_r: int, center_c: int):
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                mirror_r, mirror_c = 2 * center_r - r, 2 * center_c - c
                if 0 <= mirror_r < rows and 0 <= mirror_c < cols and grid.get_cell(mirror_r, mirror_c) in [0, 1]:
                    grid.set_cell(mirror_r, mirror_c, 8)
