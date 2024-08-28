from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Optional

def solve_e95e3d8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying the repeating pattern
    and filling in black areas with the correct pattern elements.
    
    1. Analyzes the grid to identify non-black regions
    2. Identifies the complete repeating pattern from the largest non-black region
    3. Determines the pattern offset within the grid
    4. Fills in black (0) cells with the corresponding pattern element
    5. Verifies the solution and returns the completed grid
    """
    pattern, offset = identify_pattern_and_offset(input_grid)
    if pattern is None:
        raise ValueError("No valid pattern found")
    filled_grid = fill_pattern(input_grid, pattern, offset)
    if not verify_solution(filled_grid, pattern):
        raise ValueError("Inconsistent solution")
    return filled_grid

def identify_pattern_and_offset(input_grid: ColoredGrid) -> Tuple[Optional[ColoredGrid], Tuple[int, int]]:
    rows, cols = input_grid.get_dimensions()
    non_black_regions = find_non_black_regions(input_grid)
    
    for region in sorted(non_black_regions, key=len, reverse=True):
        for height in range(1, rows // 2 + 1):
            for width in range(1, cols // 2 + 1):
                if rows % height == 0 and cols % width == 0:
                    pattern = extract_pattern(input_grid, region[0], height, width)
                    if is_valid_pattern(input_grid, pattern):
                        offset = calculate_offset(input_grid, pattern)
                        return pattern, offset
    
    return None, (0, 0)

def find_non_black_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0 and (r, c) not in visited:
                region = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] != 0:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < rows and 0 <= new_c < cols:
                                stack.append((new_r, new_c))
                regions.append(region)
    
    return regions

def extract_pattern(grid: ColoredGrid, start: Tuple[int, int], height: int, width: int) -> ColoredGrid:
    r, c = start
    return ColoredGrid(values=[
        [grid.values[(r + i) % grid.num_rows][(c + j) % grid.num_cols] for j in range(width)]
        for i in range(height)
    ])

def is_valid_pattern(grid: ColoredGrid, pattern: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    pattern_rows, pattern_cols = pattern.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0 and grid.values[r][c] != pattern.values[r % pattern_rows][c % pattern_cols]:
                return False
    return True

def calculate_offset(grid: ColoredGrid, pattern: ColoredGrid) -> Tuple[int, int]:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                return r % pattern.num_rows, c % pattern.num_cols
    return 0, 0

def fill_pattern(input_grid: ColoredGrid, pattern: ColoredGrid, offset: Tuple[int, int]) -> ColoredGrid:
    rows, cols = input_grid.get_dimensions()
    pattern_rows, pattern_cols = pattern.get_dimensions()
    offset_r, offset_c = offset
    
    new_values = [
        [pattern.values[(r + offset_r) % pattern_rows][(c + offset_c) % pattern_cols] for c in range(cols)]
        for r in range(rows)
    ]
    return ColoredGrid(values=new_values)

def verify_solution(filled_grid: ColoredGrid, pattern: ColoredGrid) -> bool:
    return is_valid_pattern(filled_grid, pattern) and 0 not in [cell for row in filled_grid.values for cell in row]
