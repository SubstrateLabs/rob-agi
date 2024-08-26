from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_2c0b0aff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 2c0b0aff challenge by identifying and extracting the largest complete pattern from the input grid.
    
    The function performs the following steps:
    1. Identifies non-black cells in the input grid
    2. Finds contiguous regions using flood fill
    3. Extracts candidate patterns from each region
    4. Validates patterns by checking if they appear completely in other regions
    5. Selects the largest valid pattern
    6. Optimizes the pattern by removing black edges
    7. Returns the optimized pattern as a compact ColoredGrid
    
    Args:
    input_grid (ColoredGrid): The input grid containing partial pattern information
    
    Returns:
    ColoredGrid: A compact grid containing the extracted largest complete pattern
    """
    non_black_cells = find_non_black_cells(input_grid)
    
    if not non_black_cells:
        return ColoredGrid(values=[[]])
    
    regions = find_contiguous_regions(input_grid, non_black_cells)
    candidate_patterns = extract_candidate_patterns(input_grid, regions)
    valid_patterns = validate_patterns(input_grid, candidate_patterns)
    best_pattern = select_best_pattern(valid_patterns)
    optimized_pattern = optimize_pattern(best_pattern)
    
    return ColoredGrid(values=optimized_pattern)

def find_non_black_cells(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    return {(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.get_cell(r, c) != 0}

def find_contiguous_regions(grid: ColoredGrid, non_black_cells: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    regions = []
    visited = set()
    
    def flood_fill(start_x: int, start_y: int) -> Set[Tuple[int, int]]:
        region = set()
        stack = [(start_x, start_y)]
        while stack:
            x, y = stack.pop()
            if (x, y) not in region and (x, y) in non_black_cells:
                region.add((x, y))
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < grid.num_rows and 0 <= new_y < grid.num_cols:
                        stack.append((new_x, new_y))
        return region
    
    for cell in non_black_cells:
        if cell not in visited:
            region = flood_fill(*cell)
            regions.append(region)
            visited.update(region)
    
    return regions

def extract_candidate_patterns(grid: ColoredGrid, regions: List[Set[Tuple[int, int]]]) -> List[Tuple[List[List[int]], Tuple[int, int]]]:
    candidate_patterns = []
    
    for region in regions:
        min_x = min(x for x, _ in region)
        max_x = max(x for x, _ in region)
        min_y = min(y for _, y in region)
        max_y = max(y for _, y in region)
        
        pattern = []
        for x in range(min_x, max_x + 1):
            row = []
            for y in range(min_y, max_y + 1):
                row.append(grid.get_cell(x, y))
            pattern.append(row)
        
        candidate_patterns.append((pattern, (max_x - min_x + 1, max_y - min_y + 1)))
    
    return candidate_patterns

def validate_patterns(grid: ColoredGrid, candidate_patterns: List[Tuple[List[List[int]], Tuple[int, int]]]) -> List[List[List[int]]]:
    valid_patterns = []
    
    for pattern, dimensions in candidate_patterns:
        if is_valid_pattern(pattern, dimensions, grid):
            valid_patterns.append(pattern)
    
    return valid_patterns

def is_valid_pattern(pattern: List[List[int]], dimensions: Tuple[int, int], grid: ColoredGrid) -> bool:
    pattern_height, pattern_width = dimensions
    for start_x in range(grid.num_rows - pattern_height + 1):
        for start_y in range(grid.num_cols - pattern_width + 1):
            if all(grid.get_cell(start_x + x, start_y + y) == pattern[x][y]
                   for x in range(pattern_height)
                   for y in range(pattern_width)):
                return True
    return False

def select_best_pattern(valid_patterns: List[List[List[int]]]) -> List[List[int]]:
    return max(valid_patterns, key=lambda p: len(p) * len(p[0]))

def optimize_pattern(pattern: List[List[int]]) -> List[List[int]]:
    # Remove black rows from top and bottom
    while pattern and all(cell == 0 for cell in pattern[0]):
        pattern = pattern[1:]
    while pattern and all(cell == 0 for cell in pattern[-1]):
        pattern = pattern[:-1]
    
    # Remove black columns from left and right
    while pattern and all(row[0] == 0 for row in pattern):
        pattern = [row[1:] for row in pattern]
    while pattern and all(row[-1] == 0 for row in pattern):
        pattern = [row[:-1] for row in pattern]
    
    return pattern
