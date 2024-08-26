from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_bb52a14b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the bb52a14b challenge by finding the most distinctive 3x3 color pattern
    in the left two-thirds of the grid and replicating it in suitable areas
    on the right third of the grid.

    1. Scan the left two-thirds for 3x3 non-black color patterns.
    2. Select the most distinctive pattern based on unique colors and repetitions.
    3. Find potential replication areas in the right third of the grid.
    4. Replicate the pattern in suitable areas, preserving existing non-black colors.
    5. Limit replications to maintain balance in the grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with replicated patterns.
    """
    # Step 1 & 2: Find the most distinctive 3x3 pattern
    pattern = find_most_distinctive_pattern(input_grid)
    
    # Step 3: Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Step 4: Find and rank potential replication areas
    replication_areas = find_replication_areas(output_grid, pattern)
    
    # Step 5: Replicate the pattern in suitable areas
    replications = 0
    for area in replication_areas:
        if replications >= 3:
            break
        if replicate_pattern(output_grid, pattern, area[0], area[1]):
            replications += 1
    
    return output_grid

def find_most_distinctive_pattern(grid: ColoredGrid) -> List[List[int]]:
    """Find the most distinctive 3x3 non-black color pattern in the left two-thirds of the grid."""
    rows, cols = grid.get_dimensions()
    best_pattern = None
    best_score = -1
    
    for r in range(rows - 2):
        for c in range(int(2 * cols / 3) - 2):
            pattern = extract_pattern(grid, r, c)
            score = calculate_distinctiveness(pattern)
            if score > best_score:
                best_score = score
                best_pattern = pattern
    
    return best_pattern if best_pattern else [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

def extract_pattern(grid: ColoredGrid, start_r: int, start_c: int) -> List[List[int]]:
    """Extract a 3x3 pattern starting from the given position."""
    return [[grid.get_cell(r, c) for c in range(start_c, start_c + 3)] for r in range(start_r, start_r + 3)]

def calculate_distinctiveness(pattern: List[List[int]]) -> int:
    """Calculate the distinctiveness score of a pattern."""
    flat_pattern = [cell for row in pattern for cell in row if cell != 0]
    unique_colors = set(flat_pattern)
    return len(unique_colors) * 10 - (len(flat_pattern) - len(unique_colors))

def find_replication_areas(grid: ColoredGrid, pattern: List[List[int]]) -> List[Tuple[int, int, int]]:
    """Find potential replication areas in the right third of the grid."""
    rows, cols = grid.get_dimensions()
    start_col = int(2 * cols / 3)
    areas = []
    for r in range(rows - 2):
        for c in range(start_col, cols - 2):
            match_score = calculate_match_score(grid, pattern, r, c)
            if match_score >= 3:
                areas.append((r, c, match_score))
    return sorted(areas, key=lambda x: (-x[2], abs(x[0] - rows/2)))  # Sort by match score (desc) and centrality

def calculate_match_score(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int) -> int:
    """Calculate the match score between the pattern and the grid area."""
    score = 0
    for r in range(3):
        for c in range(3):
            grid_value = grid.get_cell(start_r + r, start_c + c)
            pattern_value = pattern[r][c]
            if grid_value == pattern_value and grid_value != 0:
                score += 3
            elif grid_value == 0:
                score += 1
    return score

def replicate_pattern(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int) -> bool:
    """Replicate the given pattern at the specified position in the grid, preserving existing non-black colors."""
    changes_made = False
    for r in range(3):
        for c in range(3):
            grid_value = grid.get_cell(start_r + r, start_c + c)
            pattern_value = pattern[r][c]
            if grid_value == 0 and pattern_value != 0:
                grid.set_cell(start_r + r, start_c + c, pattern_value)
                changes_made = True
    return changes_made
