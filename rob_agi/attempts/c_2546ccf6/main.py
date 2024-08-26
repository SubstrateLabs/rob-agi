from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying patterns and applying vertical mirroring within sections.
    
    1. Identifies dividing lines (color 2 or 6) in the grid.
    2. Defines sections based on these dividing lines.
    3. Processes sections:
       a. Identifies the topmost non-empty section as the "source" section.
       b. Mirrors the source section to the section immediately below (if it exists).
       c. For subsequent sections, mirrors the bottom half upwards within the section.
    4. Preserves all dividing lines.
    5. Handles edge cases for grids with one or two sections.
    
    Returns a new grid with the transformed patterns.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # Find dividing lines
    h_dividers = [r for r in range(rows) if all(grid.values[r][c] in [2, 6] for c in range(cols))]
    v_dividers = [c for c in range(cols) if all(grid.values[r][c] in [2, 6] for r in range(rows))]
    
    # Define sections
    sections = []
    for i in range(len(h_dividers) - 1):
        section = (h_dividers[i] + 1, 0, h_dividers[i+1], cols)
        if not is_empty_section(grid, section):
            sections.append(section)
    
    # Process sections
    if len(sections) == 1:
        return grid  # No changes needed for single section
    elif len(sections) >= 2:
        source_section = sections[0]
        mirror_to_next_section(grid, source_section, sections[1])
        
        for section in sections[2:]:
            mirror_within_section(grid, section)
    
    return grid

def is_empty_section(grid: ColoredGrid, section: Tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = section
    return all(grid.values[r][c] == 0 for r in range(top, bottom) for c in range(left, right))

def mirror_to_next_section(grid: ColoredGrid, source: Tuple[int, int, int, int], target: Tuple[int, int, int, int]):
    src_top, _, src_bottom, _ = source
    tgt_top, tgt_left, tgt_bottom, tgt_right = target
    src_height = src_bottom - src_top
    tgt_height = tgt_bottom - tgt_top
    
    for r in range(min(src_height, tgt_height)):
        for c in range(tgt_left, tgt_right):
            if grid.values[src_bottom - 1 - r][c] not in [0, 2, 6]:
                grid.values[tgt_top + r][c] = grid.values[src_bottom - 1 - r][c]

def mirror_within_section(grid: ColoredGrid, section: Tuple[int, int, int, int]):
    top, left, bottom, right = section
    mid = (top + bottom) // 2
    for r in range(mid, bottom):
        for c in range(left, right):
            if grid.values[r][c] not in [0, 2, 6]:
                grid.values[top + (bottom - 1 - r)][c] = grid.values[r][c]
