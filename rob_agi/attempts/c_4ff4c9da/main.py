from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding and balancing sky blue (8) formations.
    
    The function identifies existing sky blue formations, expands them where possible,
    and adds new formations to create symmetry and balance across the grid. It respects
    the existing pattern of vertical stripes and horizontal lines while making these changes.
    
    Steps:
    1. Identify all sky blue formations
    2. Expand existing formations, preferably to 3x3 squares
    3. Add new formations to create symmetry
    4. Fine-tune the arrangement to achieve overall balance
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    grid = input_grid.deep_copy()
    sky_blue_formations = find_sky_blue_formations(grid)
    
    # Expand existing formations
    for formation in sky_blue_formations:
        expand_formation(grid, formation)
    
    # Add new formations for symmetry
    add_symmetric_formations(grid)
    
    # Fine-tune for balance
    balance_formations(grid)
    
    return grid

def find_sky_blue_formations(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    return grid.find_connected_regions(8)

def expand_formation(grid: ColoredGrid, formation: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    min_r = min(r for r, _ in formation)
    max_r = max(r for r, _ in formation)
    min_c = min(c for _, c in formation)
    max_c = max(c for _, c in formation)
    
    # Try to expand to a 3x3 square if possible
    for r in range(max(0, min_r - 1), min(rows, max_r + 2)):
        for c in range(max(0, min_c - 1), min(cols, max_c + 2)):
            if grid.get_cell(r, c) in [0, 1]:  # Only expand into black or blue cells
                grid.set_cell(r, c, 8)

def add_symmetric_formations(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    center_r, center_c = rows // 2, cols // 2
    
    # Check for potential symmetric positions
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                symmetric_r = 2 * center_r - r
                symmetric_c = 2 * center_c - c
                if 0 <= symmetric_r < rows and 0 <= symmetric_c < cols:
                    if grid.get_cell(symmetric_r, symmetric_c) in [0, 1]:
                        grid.set_cell(symmetric_r, symmetric_c, 8)

def balance_formations(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8:
                # Ensure formations don't break vertical stripe pattern
                if c > 0 and c < cols - 1:
                    if grid.get_cell(r, c-1) == grid.get_cell(r, c+1) == 2:
                        grid.set_cell(r, c, 2)
                # Ensure formations don't break horizontal line pattern
                if r > 0 and r < rows - 1:
                    if grid.get_cell(r-1, c) == grid.get_cell(r+1, c) == 1:
                        grid.set_cell(r, c, 1)
