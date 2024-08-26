from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

from typing import List, Tuple, Set

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) formations symmetrically.
    
    The function identifies existing sky blue formations and expands them vertically
    and horizontally while maintaining symmetry across both axes. It only replaces
    black (0) or blue (1) cells, preserving the underlying red (2) cell pattern.
    
    Steps:
    1. Identify initial sky blue formations
    2. Perform vertical mirroring
    3. Perform horizontal mirroring for specific formations
    4. Propagate edge and corner formations
    5. Expand center vertical formations
    6. Iterate until stabilization
    7. Ensure final vertical symmetry
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    center_r, center_c = rows // 2, cols // 2
    
    while True:
        previous_grid = grid.deep_copy()
        
        vertical_mirror(grid, center_r)
        horizontal_mirror(grid, center_c)
        propagate_edges(grid)
        expand_center_vertical(grid, center_c)
        ensure_vertical_symmetry(grid, center_c)
        
        if is_stable(grid, previous_grid):
            break
    
    return grid

def vertical_mirror(grid: ColoredGrid, center_r: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                mirror_r = 2 * center_r - r
                if 0 <= mirror_r < rows and grid.get_cell(mirror_r, c) in [0, 1]:
                    grid.set_cell(mirror_r, c, 8)

def horizontal_mirror(grid: ColoredGrid, center_c: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                if is_horizontal_line(grid, r, c) or is_square(grid, r, c):
                    mirror_c = 2 * center_c - c
                    if 0 <= mirror_c < cols and grid.get_cell(r, mirror_c) in [0, 1]:
                        grid.set_cell(r, mirror_c, 8)

def is_horizontal_line(grid: ColoredGrid, r: int, c: int) -> bool:
    cols = grid.get_dimensions()[1]
    return c > 0 and c < cols - 1 and grid.get_cell(r, c-1) == 8 and grid.get_cell(r, c+1) == 8

def is_square(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    if r < rows - 1 and c < cols - 1:
        return all(grid.get_cell(r+dr, c+dc) == 8 for dr in range(2) for dc in range(2))
    return False

def propagate_edges(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        if grid.get_cell(r, 0) == 8:
            grid.set_cell(r, cols-1, 8)
        if grid.get_cell(r, cols-1) == 8:
            grid.set_cell(r, 0, 8)
    for c in range(cols):
        if grid.get_cell(0, c) == 8:
            grid.set_cell(rows-1, c, 8)
        if grid.get_cell(rows-1, c) == 8:
            grid.set_cell(0, c, 8)

def expand_center_vertical(grid: ColoredGrid, center_c: int):
    rows, cols = grid.get_dimensions()
    center_range = range(center_c - cols//4, center_c + cols//4 + 1)
    for c in center_range:
        if any(grid.get_cell(r, c) == 8 for r in range(rows)):
            for r in range(rows):
                if grid.get_cell(r, c) in [0, 1]:
                    grid.set_cell(r, c, 8)

def ensure_vertical_symmetry(grid: ColoredGrid, center_c: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                mirror_c = 2 * center_c - c
                if 0 <= mirror_c < cols and grid.get_cell(r, mirror_c) in [0, 1]:
                    grid.set_cell(r, mirror_c, 8)

def is_stable(grid: ColoredGrid, previous_grid: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    return all(grid.get_cell(r, c) == previous_grid.get_cell(r, c) 
               for r in range(rows) for c in range(cols))
