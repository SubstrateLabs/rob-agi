from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying vertical mirroring within sections.
    
    1. Identifies horizontal dividing lines (color 2 or 6) in the grid.
    2. Divides the grid into top and bottom halves.
    3. For each pair of corresponding sections in top and bottom halves:
       a. Mirrors the content of the top section to the bottom section.
    4. Preserves all dividing lines.
    
    Returns a new grid with the transformed patterns.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # Find horizontal dividing lines
    h_dividers = [r for r in range(rows) if all(grid.values[r][c] in [2, 6] for c in range(cols))]
    
    # Define sections
    sections = []
    for i in range(len(h_dividers) - 1):
        section = (h_dividers[i] + 1, h_dividers[i+1] - 1)
        if not is_empty_section(grid, section, cols):
            sections.append(section)
    
    # Process sections
    n = len(sections) // 2
    for i in range(n):
        mirror_sections(grid, sections[i], sections[i+n], cols)
    
    return grid

def is_empty_section(grid: ColoredGrid, section: Tuple[int, int], cols: int) -> bool:
    top, bottom = section
    return all(grid.values[r][c] == 0 for r in range(top, bottom+1) for c in range(cols))

def mirror_sections(grid: ColoredGrid, top_section: Tuple[int, int], bottom_section: Tuple[int, int], cols: int):
    top_start, top_end = top_section
    bottom_start, bottom_end = bottom_section
    
    for r in range(top_end - top_start + 1):
        for c in range(cols):
            if grid.values[top_end - r][c] not in [0, 2, 6]:
                grid.values[bottom_start + r][c] = grid.values[top_end - r][c]
