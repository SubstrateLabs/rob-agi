from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_bb52a14b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the bb52a14b challenge by finding a 3x3 "flower" pattern in the grid and replicating it
    up to two times in suitable areas across the entire grid.

    1. Identify the flower pattern (8 in center surrounded by 4s, or 4 in center with other colors).
    2. Find potential replication areas, scoring them based on compatibility.
    3. Replicate the pattern up to two times, preserving all existing non-black colors.
    4. If no replications are possible, return the original grid unchanged.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with replicated patterns or the original if no replications were possible.
    """
    pattern = find_flower_pattern(input_grid)
    if not pattern:
        return input_grid

    output_grid = input_grid.deep_copy()
    replication_areas = find_replication_areas(output_grid, pattern)
    
    replications = 0
    for row, col, _ in replication_areas:
        if replicate_pattern(output_grid, pattern, row, col):
            replications += 1
            if replications == 2:
                break
    
    return output_grid

def find_flower_pattern(grid: ColoredGrid) -> Optional[List[List[int]]]:
    """Find the flower pattern in the grid."""
    rows, cols = grid.get_dimensions()
    for r in range(rows - 2):
        for c in range(cols - 2):
            pattern = extract_pattern(grid, r, c)
            if is_flower_pattern(pattern):
                return pattern
    return None

def is_flower_pattern(pattern: List[List[int]]) -> bool:
    """Check if the given pattern is a flower pattern."""
    center = pattern[1][1]
    if center == 8:
        return all(pattern[i][j] == 4 for i in range(3) for j in range(3) if (i, j) != (1, 1))
    elif center == 4:
        surrounding = [pattern[i][j] for i in range(3) for j in range(3) if (i, j) != (1, 1)]
        return all(color != 0 for color in surrounding) and len(set(surrounding)) <= 2
    return False

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
            elif grid_value != 0 and pattern_value != 0:
                if grid_value == pattern_value:
                    score += 2
                else:
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
