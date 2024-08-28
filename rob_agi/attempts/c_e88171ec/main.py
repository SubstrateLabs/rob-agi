from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Optional

def solve_e88171ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the e88171ec challenge by finding the largest contiguous region of black cells
    and filling it with the largest possible sky blue (8) rectangle.

    The solution follows these steps:
    1. Find all contiguous regions of black (0) cells.
    2. Select the largest black region.
    3. Find the largest possible rectangle within the largest black region.
    4. Fill the chosen rectangle with sky blue (8).

    If no suitable region is found or no rectangle fits, return the original grid unchanged.
    """
    black_regions = find_black_regions(input_grid)
    
    if not black_regions:
        return input_grid
    
    largest_region = get_largest_region(black_regions)
    if len(largest_region) < 4:
        return input_grid
    
    blue_rectangle = find_largest_rectangle(largest_region)
    
    if blue_rectangle:
        output_grid = input_grid.deep_copy()
        top, left, bottom, right = blue_rectangle
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
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

def find_largest_rectangle(region: Set[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not region:
        return None
    
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    
    heights = [0] * (max_c - min_c + 1)
    max_rectangle = (0, 0, 0, 0)
    max_area = 0
    
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if (r, c) in region:
                heights[c - min_c] += 1
            else:
                heights[c - min_c] = 0
        
        stack = []
        for i, h in enumerate(heights + [0]):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                width = i - index
                area = width * height
                if area > max_area and width % 2 == 0 and height % 2 == 0:
                    max_area = area
                    max_rectangle = (r - height + 1, index + min_c, r, index + min_c + width - 1)
                start = index
            stack.append((start, h))
    
    return max_rectangle if max_area > 0 else None
