from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_e88171ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e88171ec challenge by finding the largest contiguous region of black cells
    and filling it with the largest possible sky blue (8) rectangle.

    The solution follows these steps:
    1. Find all contiguous regions of black (0) cells.
    2. Select the largest black region.
    3. Attempt to place a 4x4 sky blue rectangle in the largest region.
    4. If a 4x4 doesn't fit, attempt to place a 2x2 sky blue rectangle.
    5. Place the rectangle in the bottom-right most position possible within the region.
    6. Fill the chosen area with sky blue (8).

    If no suitable region is found or no rectangle fits, return the original grid unchanged.
    """
    black_regions = find_black_regions(input_grid)
    
    if not black_regions:
        return input_grid
    
    largest_region = get_largest_region(black_regions)
    if len(largest_region) < 4:
        return input_grid
    
    bounding_box = get_bounding_box(largest_region)
    blue_rectangle = find_rectangle_position(largest_region, bounding_box)
    
    if blue_rectangle:
        output_grid = input_grid.deep_copy()
        size, (top, left) = blue_rectangle
        for r in range(top, top + size):
            for c in range(left, left + size):
                output_grid.set_cell(r, c, 8)
        return output_grid
    
    return input_grid

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

def find_rectangle_position(region: Set[Tuple[int, int]], bounding_box: Tuple[int, int, int, int]) -> Optional[Tuple[int, Tuple[int, int]]]:
    min_r, max_r, min_c, max_c = bounding_box
    
    for size in [4, 2]:
        for r in range(max_r, min_r - 1, -1):
            for c in range(max_c, min_c - 1, -1):
                if r - size + 1 < min_r or c - size + 1 < min_c:
                    continue
                if all((rr, cc) in region for rr in range(r - size + 1, r + 1) for cc in range(c - size + 1, c + 1)):
                    return size, (r - size + 1, c - size + 1)
    
    return None
