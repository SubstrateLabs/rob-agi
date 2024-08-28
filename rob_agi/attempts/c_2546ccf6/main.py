from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying vertical mirroring within sections.
    
    1. Identifies horizontal dividing lines (color 2 or 6) in the grid.
    2. Divides the grid into vertical sections based on these dividing lines.
    3. For each vertical section:
       a. Analyzes the pattern complexity in the top and bottom halves.
       b. Mirrors the more complex pattern to the less complex half.
       c. If one half is empty, mirrors the non-empty half's pattern.
    4. Preserves all dividing lines.
    
    Returns a new grid with the transformed patterns.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    dividing_lines = find_dividing_lines(grid)
    sections = create_sections(dividing_lines, rows)
    
    for section in sections:
        mirror_vertical_section(grid, section, cols)
    
    return grid

def find_dividing_lines(grid: ColoredGrid) -> List[int]:
    rows, cols = grid.get_dimensions()
    return [r for r in range(rows) if all(grid.values[r][c] in [2, 6] for c in range(cols))]

def create_sections(dividing_lines: List[int], rows: int) -> List[Tuple[int, int]]:
    sections = []
    start = 0
    for line in dividing_lines:
        if line > start:
            sections.append((start, line - 1))
        start = line + 1
    if start < rows:
        sections.append((start, rows - 1))
    return sections

def mirror_vertical_section(grid: ColoredGrid, section: Tuple[int, int], cols: int):
    start, end = section
    mid = (start + end) // 2
    
    top_half = (start, mid)
    bottom_half = (mid + 1, end)
    
    top_complexity = analyze_complexity(grid, top_half, cols)
    bottom_complexity = analyze_complexity(grid, bottom_half, cols)
    
    if top_complexity >= bottom_complexity:
        mirror_half(grid, top_half, bottom_half, cols)
    else:
        mirror_half(grid, bottom_half, top_half, cols)

def analyze_complexity(grid: ColoredGrid, half: Tuple[int, int], cols: int) -> int:
    start, end = half
    non_zero_count = sum(1 for r in range(start, end + 1) for c in range(cols) if grid.values[r][c] not in [0, 2, 6])
    unique_colors = len(set(grid.values[r][c] for r in range(start, end + 1) for c in range(cols) if grid.values[r][c] not in [0, 2, 6]))
    return non_zero_count + unique_colors

def mirror_half(grid: ColoredGrid, source: Tuple[int, int], target: Tuple[int, int], cols: int):
    source_start, source_end = source
    target_start, target_end = target
    
    for r in range(source_end - source_start + 1):
        for c in range(cols):
            if grid.values[source_end - r][c] not in [0, 2, 6]:
                grid.values[target_start + r][c] = grid.values[source_end - r][c]
