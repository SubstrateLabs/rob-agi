from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_e88171ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e88171ec challenge by finding the largest contiguous region of black cells
    and filling it with the largest possible centered, even-dimensioned sky blue (8) rectangle.

    The solution follows these steps:
    1. Find all contiguous regions of black (0) cells.
    2. Select the largest black region.
    3. Determine the dimensions of the sky blue rectangle (even numbers, max 4x4).
    4. Optimize the placement of the sky blue rectangle within the black region.
    5. Fill the chosen area with sky blue (8).

    If no suitable region is found, return the original grid unchanged.
    """
    output_grid = input_grid.deep_copy()
    black_regions = find_black_regions(input_grid)
    
    if not black_regions:
        return output_grid
    
    largest_region = get_largest_region(black_regions)
    if len(largest_region) < 4:
        return output_grid
    
    bounding_box = get_bounding_box(largest_region)
    blue_rectangle = calculate_blue_rectangle(largest_region, bounding_box)
    
    if blue_rectangle:
        width, height, top_left = blue_rectangle
        for r in range(top_left[0], top_left[0] + height):
            for c in range(top_left[1], top_left[1] + width):
                output_grid.set_cell(r, c, 8)
    
    return output_grid

def find_black_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0 and (r, c) not in visited:
                region = set()
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited:
                        visited.add((curr_r, curr_c))
                        region.add((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < rows and 0 <= new_c < cols and grid.get_cell(new_r, new_c) == 0:
                                stack.append((new_r, new_c))
                regions.append(region)
    return regions

def get_largest_region(regions: List[Set[Tuple[int, int]]]) -> Set[Tuple[int, int]]:
    return max(regions, key=len)

def get_bounding_box(region: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    return min_r, max_r, min_c, max_c

def calculate_blue_rectangle(region: Set[Tuple[int, int]], bounding_box: Tuple[int, int, int, int]) -> Tuple[int, int, Tuple[int, int]]:
    min_r, max_r, min_c, max_c = bounding_box
    center_r, center_c = (min_r + max_r) // 2, (min_c + max_c) // 2
    
    for size in [4, 2]:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                top = center_r - size // 2 + dr
                left = center_c - size // 2 + dc
                if all((r, c) in region for r in range(top, top + size) for c in range(left, left + size)):
                    return size, size, (top, left)
    
    return None
