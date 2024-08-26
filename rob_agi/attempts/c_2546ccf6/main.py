from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2546ccf6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying patterns and applying transformations.
    
    1. Identifies dividing lines and sections in the grid.
    2. Locates the focus section with the most non-zero, non-divider elements.
    3. Extracts the pattern from the focus section.
    4. Generates transformed patterns (flipped and rotated).
    5. Applies transformed patterns to adjacent sections.
    6. Clears remaining elements in non-focus, non-adjacent sections.
    7. Preserves all dividing lines.
    
    Returns a new grid with the transformed pattern.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    # Find vertical dividing lines
    dividers = [i for i in range(cols) if grid.values[0][i] in [2, 6]]
    if len(dividers) < 2:
        return grid  # Not enough dividers, return original grid
    
    # Identify sections
    sections = [(dividers[i], dividers[i+1]) for i in range(len(dividers)-1)]
    
    # Find focus section
    focus_section = max(sections, key=lambda s: sum(1 for r in range(rows) for c in range(s[0], s[1]) if grid.values[r][c] not in [0, 2, 6]))
    
    # Extract pattern from focus section
    pattern = extract_pattern(grid, focus_section, rows)
    
    # Generate transformed patterns
    flipped_vertical = flip_pattern_vertical(pattern)
    flipped_horizontal = flip_pattern_horizontal(pattern)
    rotated = rotate_pattern_180(pattern)
    
    # Apply transformed patterns to adjacent sections
    for section in sections:
        if section == focus_section:
            continue
        if section[0] < focus_section[0]:
            apply_pattern(grid, section, flipped_horizontal, rows)
        elif section[1] > focus_section[1]:
            apply_pattern(grid, section, flipped_horizontal, rows)
    
    # Clear remaining elements in non-focus, non-adjacent sections
    for section in sections:
        if section != focus_section and not is_adjacent(section, focus_section):
            clear_section(grid, section, rows)
    
    return grid

def extract_pattern(grid: ColoredGrid, section: Tuple[int, int], rows: int) -> List[List[int]]:
    return [[grid.values[r][c] for c in range(section[0], section[1])] for r in range(rows)]

def flip_pattern_vertical(pattern: List[List[int]]) -> List[List[int]]:
    return pattern[::-1]

def flip_pattern_horizontal(pattern: List[List[int]]) -> List[List[int]]:
    return [row[::-1] for row in pattern]

def rotate_pattern_180(pattern: List[List[int]]) -> List[List[int]]:
    return [row[::-1] for row in pattern[::-1]]

def apply_pattern(grid: ColoredGrid, section: Tuple[int, int], pattern: List[List[int]], rows: int):
    for r in range(rows):
        for c, val in enumerate(pattern[r % len(pattern)]):
            if grid.values[r][section[0] + c] not in [2, 6]:
                grid.values[r][section[0] + c] = val

def clear_section(grid: ColoredGrid, section: Tuple[int, int], rows: int):
    for r in range(rows):
        for c in range(section[0], section[1]):
            if grid.values[r][c] not in [2, 6]:
                grid.values[r][c] = 0

def is_adjacent(section1: Tuple[int, int], section2: Tuple[int, int]) -> bool:
    return section1[1] == section2[0] or section2[1] == section1[0]
