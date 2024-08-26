from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_bb52a14b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the bb52a14b challenge by finding the most frequent 3x3 color pattern
    in the left two-thirds of the grid and replicating it in suitable areas
    on the right third of the grid.

    1. Scan the left two-thirds for 3x3 non-black color patterns.
    2. Select the most frequent pattern with the most distinctive colors.
    3. Find potential replication areas in the right third of the grid.
    4. Replicate the pattern in suitable areas, preserving existing matches.
    5. Maintain original scattered colors outside the replicated areas.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with replicated patterns.
    """
    # Step 1: Find the most frequent 3x3 pattern
    pattern, _ = find_most_frequent_pattern(input_grid)
    
    # Step 2: Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Step 3: Find and rank potential replication areas
    replication_areas = find_replication_areas(output_grid, pattern)
    
    # Step 4: Replicate the pattern in suitable areas
    for area in replication_areas[:3]:  # Limit to top 3 areas
        replicate_pattern(output_grid, pattern, area[0], area[1])
    
    return output_grid

def find_most_frequent_pattern(grid: ColoredGrid) -> Tuple[List[List[int]], int]:
    """Find the most frequent 3x3 non-black color pattern in the left two-thirds of the grid."""
    rows, cols = grid.get_dimensions()
    patterns = {}
    for r in range(rows - 2):
        for c in range(int(2 * cols / 3) - 2):
            pattern = extract_pattern(grid, r, c)
            if pattern and not all(cell == 0 for row in pattern for cell in row):
                pattern_tuple = tuple(tuple(row) for row in pattern)
                patterns[pattern_tuple] = patterns.get(pattern_tuple, 0) + 1
    
    if not patterns:
        return [], 0
    
    most_frequent = max(patterns, key=patterns.get)
    return [list(row) for row in most_frequent], patterns[most_frequent]

def extract_pattern(grid: ColoredGrid, start_r: int, start_c: int) -> List[List[int]]:
    """Extract a 3x3 pattern starting from the given position."""
    pattern = []
    for r in range(start_r, start_r + 3):
        row = []
        for c in range(start_c, start_c + 3):
            row.append(grid.get_cell(r, c))
        pattern.append(row)
    return pattern

def find_replication_areas(grid: ColoredGrid, pattern: List[List[int]]) -> List[Tuple[int, int, int]]:
    """Find potential replication areas in the right third of the grid."""
    rows, cols = grid.get_dimensions()
    start_col = int(2 * cols / 3)
    areas = []
    for r in range(rows - 2):
        for c in range(start_col, cols - 2):
            match_count = count_matches(grid, pattern, r, c)
            if match_count > 0:
                areas.append((r, c, match_count))
    return sorted(areas, key=lambda x: (-x[2], abs(x[0] - rows/2)))  # Sort by match count (desc) and centrality

def count_matches(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int) -> int:
    """Count the number of matching cells between the pattern and the grid area."""
    match_count = 0
    for r in range(3):
        for c in range(3):
            if grid.get_cell(start_r + r, start_c + c) == pattern[r][c]:
                match_count += 1
    return match_count

def replicate_pattern(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int):
    """Replicate the given pattern at the specified position in the grid, preserving existing matches."""
    for r in range(3):
        for c in range(3):
            if grid.get_cell(start_r + r, start_c + c) != pattern[r][c]:
                grid.set_cell(start_r + r, start_c + c, pattern[r][c])
