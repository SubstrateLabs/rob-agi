from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying vertical mirroring within sections.
    
    1. Identifies horizontal dividing lines (color 2 or 6) in the grid.
    2. Divides the grid into sections based on these dividing lines.
    3. Pairs corresponding top and bottom sections.
    4. For each pair of sections:
       a. If the top section contains a non-zero, non-divider pattern:
          - Mirrors the pattern vertically from the top section to the bottom section.
       b. If the top section is empty, leaves both sections unchanged.
    5. Preserves all dividing lines and unaffected sections.
    
    Returns a new grid with the transformed patterns.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    dividing_lines = find_dividing_lines(grid)
    sections = create_sections(dividing_lines, rows)
    
    n = len(sections) // 2
    for i in range(n):
        top_section = sections[i]
        bottom_section = sections[-(i+1)]
        if not is_empty_section(grid, top_section, cols):
            mirror_section(grid, top_section, bottom_section, cols)
    
    return grid

def find_dividing_lines(grid: ColoredGrid) -> List[int]:
    rows, cols = grid.get_dimensions()
    return [r for r in range(rows) if all(grid.values[r][c] in [2, 6] for c in range(cols))]

def create_sections(dividing_lines: List[int], rows: int) -> List[Tuple[int, int]]:
    sections = []
    for i in range(len(dividing_lines) - 1):
        sections.append((dividing_lines[i] + 1, dividing_lines[i+1] - 1))
    return sections

def is_empty_section(grid: ColoredGrid, section: Tuple[int, int], cols: int) -> bool:
    start, end = section
    return all(grid.values[r][c] == 0 for r in range(start, end+1) for c in range(cols))

def mirror_section(grid: ColoredGrid, top_section: Tuple[int, int], bottom_section: Tuple[int, int], cols: int):
    top_start, top_end = top_section
    bottom_start, bottom_end = bottom_section
    
    for r in range(top_end - top_start + 1):
        for c in range(cols):
            if grid.values[top_end - r][c] not in [0, 2, 6]:
                grid.values[bottom_start + r][c] = grid.values[top_end - r][c]
