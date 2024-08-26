from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from typing import List, Tuple, Set

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by mirroring sky blue (8) formations across both horizontal and vertical axes.
    
    The function identifies existing sky blue formations and creates exact mirrors of these formations
    while preserving the underlying stripe pattern of the grid. It only replaces black (0) or blue (1) cells
    when creating new formations, ensuring that the existing pattern is maintained.
    
    Steps:
    1. Identify all sky blue formations
    2. Calculate mirror positions for each formation
    3. Create mirrored formations where possible
    4. Preserve the existing stripe pattern
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    grid = input_grid.deep_copy()
    sky_blue_formations = find_sky_blue_formations(grid)
    center_r, center_c = grid.get_dimensions()[0] // 2, grid.get_dimensions()[1] // 2
    
    for formation in sky_blue_formations:
        mirror_formation(grid, formation, center_r, center_c)
    
    return grid

def find_sky_blue_formations(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    return [set(formation) for formation in grid.find_connected_regions(8)]

def mirror_formation(grid: ColoredGrid, formation: Set[Tuple[int, int]], center_r: int, center_c: int):
    rows, cols = grid.get_dimensions()
    
    for r, c in formation:
        # Mirror horizontally
        mirror_h = (r, 2 * center_c - c)
        if 0 <= mirror_h[1] < cols and grid.get_cell(*mirror_h) in [0, 1]:
            grid.set_cell(*mirror_h, 8)
        
        # Mirror vertically
        mirror_v = (2 * center_r - r, c)
        if 0 <= mirror_v[0] < rows and grid.get_cell(*mirror_v) in [0, 1]:
            grid.set_cell(*mirror_v, 8)
        
        # Mirror diagonally
        mirror_d = (2 * center_r - r, 2 * center_c - c)
        if 0 <= mirror_d[0] < rows and 0 <= mirror_d[1] < cols and grid.get_cell(*mirror_d) in [0, 1]:
            grid.set_cell(*mirror_d, 8)
