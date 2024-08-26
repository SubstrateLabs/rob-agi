from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_e88171ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e88171ec challenge by finding the largest contiguous region of black cells
    and filling it with a centered, even-dimensioned sky blue (8) rectangle.

    The solution follows these steps:
    1. Find all contiguous regions of black (0) cells.
    2. Select the largest black region.
    3. Determine the dimensions of the sky blue rectangle (even numbers, max 4x4).
    4. Calculate the placement of the sky blue rectangle within the black region.
    5. Fill the chosen area with sky blue (8).

    If no suitable region is found, return the original grid unchanged.
    """
    output_grid = input_grid.deep_copy()
    black_regions = find_black_regions(input_grid)
    
    if not black_regions:
        return output_grid
    
    largest_region = max(black_regions, key=len)
    if len(largest_region) < 4:
        return output_grid
    
    fill_area = determine_fill_area(largest_region)
    
    for r, c in fill_area:
        output_grid.set_cell(r, c, 8)
    
    return output_grid

def find_black_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0 and (r, c) not in visited:
                region = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < rows and 0 <= new_c < cols and grid.get_cell(new_r, new_c) == 0:
                                stack.append((new_r, new_c))
                regions.append(region)
    return regions

def determine_fill_area(region: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    fill_height = min(4, (height // 2) * 2)
    fill_width = min(4, (width // 2) * 2)
    
    center_r = (min_r + max_r) // 2
    center_c = (min_c + max_c) // 2
    
    top = center_r - fill_height // 2
    left = center_c - fill_width // 2
    
    fill_area = []
    for r in range(top, top + fill_height):
        for c in range(left, left + fill_width):
            if (r, c) in region:
                fill_area.append((r, c))
    
    return fill_area
