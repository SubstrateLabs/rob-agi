from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_bb52a14b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the bb52a14b challenge by finding the most distinctive 3x3 color pattern
    containing yellow (4) in the grid and replicating it exactly up to two times
    in suitable areas across the entire grid.

    1. Find the distinctive yellow-containing 3x3 pattern with the most non-black colors.
    2. Identify potential replication areas across the entire grid.
    3. Replicate the pattern exactly up to two times, preserving all existing colors.
    4. If no replications are possible, return the original grid unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with replicated patterns or the original if no replications were possible.
    """
    pattern = find_distinctive_pattern(input_grid)
    replication_areas = find_replication_areas(input_grid, pattern)
    
    output_grid = input_grid.deep_copy()
    replications = 0
    
    for row, col, _ in replication_areas:
        if replicate_pattern(output_grid, pattern, row, col):
            replications += 1
            if replications == 2:
                break
    
    return output_grid

def find_distinctive_pattern(grid: ColoredGrid) -> List[List[int]]:
    """Find the most distinctive 3x3 color pattern containing yellow (4) in the grid."""
    rows, cols = grid.get_dimensions()
    best_pattern = None
    best_score = -1
    
    for r in range(rows - 2):
        for c in range(cols - 2):
            pattern = extract_pattern(grid, r, c)
            if 4 in [cell for row in pattern for cell in row]:
                score = sum(1 for cell in [cell for row in pattern for cell in row] if cell != 0)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
    
    return best_pattern if best_pattern else [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

def extract_pattern(grid: ColoredGrid, start_r: int, start_c: int) -> List[List[int]]:
    """Extract a 3x3 pattern starting from the given position."""
    return [[grid.get_cell(r, c) for c in range(start_c, start_c + 3)] for r in range(start_r, start_r + 3)]

def find_replication_areas(grid: ColoredGrid, pattern: List[List[int]]) -> List[Tuple[int, int, float]]:
    """Find potential replication areas across the entire grid."""
    rows, cols = grid.get_dimensions()
    areas = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            score = calculate_replication_score(grid, pattern, r, c)
            if score > 0:
                areas.append((r, c, score))
    return sorted(areas, key=lambda x: -x[2])  # Sort by score in descending order

def calculate_replication_score(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int) -> float:
    """Calculate the replication score for a potential area."""
    score = 0
    for r in range(3):
        for c in range(3):
            grid_value = grid.get_cell(start_r + r, start_c + c)
            pattern_value = pattern[r][c]
            if grid_value == 0 and pattern_value != 0:
                score += 1
            elif grid_value != 0 and pattern_value != 0 and grid_value != pattern_value:
                return -1  # Area is not suitable for replication
    return score

def replicate_pattern(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int) -> bool:
    """Replicate the given pattern exactly at the specified position in the grid."""
    for r in range(3):
        for c in range(3):
            grid_value = grid.get_cell(start_r + r, start_c + c)
            pattern_value = pattern[r][c]
            if grid_value != 0 and pattern_value != grid_value:
                return False  # Cannot replicate without overwriting non-black cells
    
    for r in range(3):
        for c in range(3):
            grid.set_cell(start_r + r, start_c + c, pattern[r][c])
    
    return True
