from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4852f2fa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the largest connected region of sky blue (8) squares
    and the number of yellow (4) squares.
    
    1. Count yellow (4) squares in the input grid.
    2. Find the largest connected region of sky blue (8) squares.
    3. Transform the largest region into a 3xN pattern.
    4. Adjust the pattern to fit the width determined by yellow squares count.
    5. Create the final 3xN output grid where N = yellow_count * 3.
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    yellow_count = sum(row.count(4) for row in input_grid.values)
    output_width = yellow_count * 3
    
    largest_region = find_largest_sky_blue_region(input_grid)
    pattern = transform_to_pattern(largest_region)
    adjusted_pattern = adjust_pattern(pattern, output_width)
    
    output = [[0 for _ in range(output_width)] for _ in range(3)]
    for i in range(output_width):
        for j in range(3):
            output[j][i] = adjusted_pattern[j][i % len(adjusted_pattern[0])]
    
    return ColoredGrid(values=output)

def find_largest_sky_blue_region(grid: ColoredGrid) -> List[List[int]]:
    visited = set()
    largest_region = []
    rows, cols = grid.get_dimensions()
    
    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid.values[r][c] != 8:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(r + dr, c + dc))
        return region
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                region = dfs(r, c)
                if len(region) > len(largest_region):
                    largest_region = region
    
    min_r = min(r for r, _ in largest_region)
    max_r = max(r for r, _ in largest_region)
    min_c = min(c for _, c in largest_region)
    max_c = max(c for _, c in largest_region)
    
    extracted_region = [[0 for _ in range(min_c, max_c + 1)] for _ in range(min_r, max_r + 1)]
    for r, c in largest_region:
        extracted_region[r - min_r][c - min_c] = 8
    
    return extracted_region

def transform_to_pattern(region: List[List[int]]) -> List[List[int]]:
    width = len(region[0])
    pattern = [[0 for _ in range(width)] for _ in range(3)]
    for c in range(width):
        if any(region[r][c] == 8 for r in range(len(region))):
            for r in range(3):
                pattern[r][c] = 8
    return pattern

def adjust_pattern(pattern: List[List[int]], target_width: int) -> List[List[int]]:
    pattern_width = len(pattern[0])
    if pattern_width > target_width:
        return [row[:target_width] for row in pattern]
    elif pattern_width < target_width:
        repetitions = target_width // pattern_width + 1
        return [row * repetitions for row in pattern]
    return pattern
