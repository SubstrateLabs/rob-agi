from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying patterns and applying transformations.
    
    1. Identifies dividing lines and sections in the grid.
    2. Analyzes each section for non-empty patterns.
    3. Applies vertical mirroring within non-empty sections.
    4. Mirrors patterns horizontally and vertically to adjacent sections.
    5. Clears unused sections.
    6. Preserves all dividing lines.
    
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
    
    # Analyze and transform sections
    for section in sections:
        if is_section_non_empty(grid, section):
            mirror_within_section(grid, section)
            mirror_to_adjacent_sections(grid, section, sections)
    
    # Clear unused sections
    for section in sections:
        if not is_section_non_empty(grid, section) and not has_adjacent_non_empty(grid, section, sections):
            clear_section(grid, section)
    
    return grid

def is_section_non_empty(grid: ColoredGrid, section: Tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = section
    return any(grid.values[r][c] not in [0, 2, 6] for r in range(top, bottom) for c in range(left, right))

def mirror_within_section(grid: ColoredGrid, section: Tuple[int, int, int, int]):
    top, left, bottom, right = section
    mid = (top + bottom) // 2
    for r in range(top, mid):
        for c in range(left, right):
            if grid.values[r][c] not in [2, 6]:
                grid.values[bottom - 1 - (r - top)][c] = grid.values[r][c]

def mirror_to_adjacent_sections(grid: ColoredGrid, section: Tuple[int, int, int, int], all_sections: List[Tuple[int, int, int, int]]):
    for adj_section in all_sections:
        if is_adjacent_horizontal(section, adj_section):
            mirror_horizontal(grid, section, adj_section)
        elif is_adjacent_vertical(section, adj_section):
            mirror_vertical(grid, section, adj_section)

def is_adjacent_horizontal(section1: Tuple[int, int, int, int], section2: Tuple[int, int, int, int]) -> bool:
    return section1[2] == section2[2] and section1[3] == section2[1]

def is_adjacent_vertical(section1: Tuple[int, int, int, int], section2: Tuple[int, int, int, int]) -> bool:
    return section1[1] == section2[1] and section1[2] == section2[0]

def mirror_horizontal(grid: ColoredGrid, source: Tuple[int, int, int, int], target: Tuple[int, int, int, int]):
    s_top, s_left, s_bottom, s_right = source
    t_top, t_left, t_bottom, t_right = target
    for r in range(s_top, s_bottom):
        for c in range(s_left, s_right):
            if grid.values[r][c] not in [2, 6]:
                grid.values[r][t_right - 1 - (c - s_left)] = grid.values[r][c]

def mirror_vertical(grid: ColoredGrid, source: Tuple[int, int, int, int], target: Tuple[int, int, int, int]):
    s_top, s_left, s_bottom, s_right = source
    t_top, t_left, t_bottom, t_right = target
    for r in range(s_top, s_bottom):
        for c in range(s_left, s_right):
            if grid.values[r][c] not in [2, 6]:
                grid.values[t_bottom - 1 - (r - s_top)][c] = grid.values[r][c]

def clear_section(grid: ColoredGrid, section: Tuple[int, int, int, int]):
    top, left, bottom, right = section
    for r in range(top, bottom):
        for c in range(left, right):
            if grid.values[r][c] not in [2, 6]:
                grid.values[r][c] = 0

def has_adjacent_non_empty(grid: ColoredGrid, section: Tuple[int, int, int, int], all_sections: List[Tuple[int, int, int, int]]) -> bool:
    return any(
        (is_adjacent_horizontal(section, adj) or is_adjacent_vertical(section, adj)) and is_section_non_empty(grid, adj)
        for adj in all_sections if adj != section
    )
