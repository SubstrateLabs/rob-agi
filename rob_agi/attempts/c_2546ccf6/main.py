from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying patterns and applying vertical mirroring within sections.
    
    1. Identifies dividing lines (color 2 or 6) in the grid.
    2. Defines rectangular sections based on these dividing lines.
    3. For each non-empty section, mirrors the top half pattern to the bottom half.
    4. Preserves all dividing lines.
    5. Leaves empty sections unchanged.
    
    Returns a new grid with the transformed patterns.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # Find dividing lines
    h_dividers = [r for r in range(rows) if all(grid.values[r][c] in [2, 6] for c in range(cols))]
    v_dividers = [c for c in range(cols) if all(grid.values[r][c] in [2, 6] for r in range(rows))]
    
    # Identify sections
    sections = []
    for i in range(len(h_dividers) - 1):
        for j in range(len(v_dividers) - 1):
            sections.append((h_dividers[i], v_dividers[j], h_dividers[i+1], v_dividers[j+1]))
    
    # Process each section
    for section in sections:
        mirror_within_section(grid, section)
    
    return grid

def mirror_within_section(grid: ColoredGrid, section: Tuple[int, int, int, int]):
    top, left, bottom, right = section
    mid = (top + bottom) // 2
    for r in range(top, mid):
        for c in range(left, right):
            if grid.values[r][c] not in [0, 2, 6]:
                grid.values[bottom - 1 - (r - top)][c] = grid.values[r][c]

# Remove unused functions
