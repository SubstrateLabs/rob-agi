from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def identify_regions(grid: ColoredGrid) -> List[Dict]:
    regions = []
    visited = set()
    
    def flood_fill(r: int, c: int, color: int) -> Set[Tuple[int, int]]:
        region = set()
        queue = deque([(r, c)])
        while queue:
            curr_r, curr_c = queue.popleft()
            if (curr_r, curr_c) not in visited and 0 <= curr_r < grid.num_rows and 0 <= curr_c < grid.num_cols and grid.values[curr_r][curr_c] == color:
                visited.add((curr_r, curr_c))
                region.add((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    queue.append((curr_r + dr, curr_c + dc))
        return region

    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited:
                color = grid.values[r][c]
                region = flood_fill(r, c, color)
                if region:
                    regions.append({"color": color, "cells": region, "size": len(region)})
    
    return regions

def expand_region(grid: ColoredGrid, region: Dict) -> None:
    color = region["color"]
    cells = region["cells"]
    if color == 3:  # Green areas are preserved
        return
    
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    
    size = max(max_r - min_r + 1, max_c - min_c + 1)
    
    if size == 2:
        for r in range(min_r - 1, max_r + 2):
            for c in range(min_c - 1, max_c + 2):
                if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                    if min_r <= r <= max_r and min_c <= c <= max_c:
                        grid.values[r][c] = color
                    else:
                        if grid.values[r][c] != 3:  # Don't overwrite green
                            grid.values[r][c] = color
    elif size == 3:
        for r in range(min_r - 1, max_r + 2):
            for c in range(min_c - 1, max_c + 2):
                if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols:
                    if grid.values[r][c] != 3:  # Don't overwrite green
                        grid.values[r][c] = color

def connect_regions(grid: ColoredGrid) -> None:
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            color = grid.values[r][c]
            if color != 0 and color != 3:  # Not black or green
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                        if grid.values[nr][nc] == color:
                            # Fill the space between with the same color
                            mid_r, mid_c = (r + nr) // 2, (c + nc) // 2
                            if grid.values[mid_r][mid_c] != 3:  # Don't overwrite green
                                grid.values[mid_r][mid_c] = color

def solve_3ed85e70(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following rules:
    1. Identify color regions in the grid.
    2. Expand 2x2 colored regions to 4x4 with borders of the same color.
    3. Expand 3x3 colored regions to 5x5 without changing their color.
    4. Preserve green (3) areas without changes.
    5. Connect adjacent regions of the same color.
    6. Repeat until no more changes can be made.
    7. Resolve conflicts by prioritizing green areas and larger regions.
    8. Maintain the internal structure of regions larger than 3x3.
    """
    grid = input_grid.deep_copy()
    
    while True:
        original = grid.deep_copy()
        regions = identify_regions(grid)
        
        for region in regions:
            if region["size"] <= 9:  # Only expand regions up to 3x3
                expand_region(grid, region)
        
        connect_regions(grid)
        
        if grid.values == original.values:
            break
    
    return grid
