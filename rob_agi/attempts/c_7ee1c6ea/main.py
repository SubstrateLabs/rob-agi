from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import random

def solve_7ee1c6ea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by balancing colors within connected regions.
    
    1. Identifies connected regions (excluding gray and black).
    2. For each region, calculates ideal color frequencies.
    3. Iteratively balances colors within regions by changing squares' colors.
    4. Preserves black (0) and gray (5) squares.
    5. Avoids creating 2x2 squares of the same color.
    6. Repeats the process until stability is reached or max iterations hit.
    
    Returns the transformed grid.
    """
    new_grid = input_grid.deep_copy()
    max_iterations = 10
    for _ in range(max_iterations):
        regions = find_all_regions(new_grid)
        if not balance_colors_in_regions(new_grid, regions):
            break
    return new_grid

def find_all_regions(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] not in [0, 5]:
                region = find_connected_region(grid, r, c, grid.values[r][c])
                regions.append(region)
                visited.update(region)
    return regions

def find_connected_region(grid: ColoredGrid, r: int, c: int, color: int) -> Set[Tuple[int, int]]:
    region = set()
    stack = [(r, c)]
    rows, cols = grid.get_dimensions()
    while stack:
        curr_r, curr_c = stack.pop()
        if (curr_r, curr_c) in region:
            continue
        if 0 <= curr_r < rows and 0 <= curr_c < cols and grid.values[curr_r][curr_c] == color:
            region.add((curr_r, curr_c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((curr_r + dr, curr_c + dc))
    return region

def balance_colors_in_regions(grid: ColoredGrid, regions: List[Set[Tuple[int, int]]]) -> bool:
    changed = False
    for region in regions:
        color_freq = defaultdict(int)
        for r, c in region:
            color_freq[grid.values[r][c]] += 1
        
        ideal_freq = len(region) / len(color_freq)
        donors = [color for color, freq in color_freq.items() if freq > ideal_freq]
        receivers = [color for color, freq in color_freq.items() if freq < ideal_freq]
        
        while donors and receivers:
            donor = donors[0]
            receiver = receivers[0]
            donor_squares = [sq for sq in region if grid.values[sq[0]][sq[1]] == donor]
            random.shuffle(donor_squares)
            
            for sq in donor_squares:
                if is_valid_change(grid, sq, receiver):
                    grid.values[sq[0]][sq[1]] = receiver
                    color_freq[donor] -= 1
                    color_freq[receiver] += 1
                    changed = True
                    break
            
            if color_freq[donor] <= ideal_freq:
                donors.pop(0)
            if color_freq[receiver] >= ideal_freq:
                receivers.pop(0)
    
    return changed

def is_valid_change(grid: ColoredGrid, sq: Tuple[int, int], new_color: int) -> bool:
    r, c = sq
    rows, cols = grid.get_dimensions()
    if grid.values[r][c] in [0, 5]:
        return False
    
    for dr, dc in [(0, 1), (1, 0), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if all(grid.values[r+i][c+j] == new_color for i in range(2) for j in range(2) if 0 <= r+i < rows and 0 <= c+j < cols):
                return False
    
    return True
