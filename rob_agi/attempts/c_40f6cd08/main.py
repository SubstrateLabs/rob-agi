from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def analyze_top_left(grid: ColoredGrid) -> List[Tuple[int, int, int, int, int]]:
    """Analyze the top-left quadrant and return a list of colored regions."""
    regions = []
    for i in range(14):
        for j in range(14):
            color = grid.values[i][j]
            if color != 0:
                regions.append((color, i, j, i, j))
    
    # Merge adjacent regions of the same color
    merged = True
    while merged:
        merged = False
        for i, r1 in enumerate(regions):
            for j, r2 in enumerate(regions[i+1:], i+1):
                if r1[0] == r2[0] and (
                    (r1[1] <= r2[3]+1 and r2[1] <= r1[3]+1 and r1[2] <= r2[4]+1 and r2[2] <= r1[4]+1)
                ):
                    new_region = (r1[0], min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3]), max(r1[4], r2[4]))
                    regions[i] = new_region
                    regions.pop(j)
                    merged = True
                    break
            if merged:
                break
    
    return regions

def is_quadrant_non_empty(grid: ColoredGrid, top: int, left: int, bottom: int, right: int) -> bool:
    """Check if a quadrant is non-empty."""
    return any(grid.values[i][j] != 0 for i in range(top, bottom+1) for j in range(left, right+1))

def replicate_pattern(source: ColoredGrid, target: ColoredGrid, regions: List[Tuple[int, int, int, int, int]], 
                      src_top: int, src_left: int, tgt_top: int, tgt_left: int):
    """Replicate the pattern from source to target quadrant."""
    for color, top, left, bottom, right in regions:
        rel_top, rel_left = top - src_top, left - src_left
        rel_bottom, rel_right = bottom - src_top, right - src_left
        for i in range(rel_top, rel_bottom + 1):
            for j in range(rel_left, rel_right + 1):
                target.values[tgt_top + i][tgt_left + j] = color

def solve_40f6cd08(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by analyzing the top-left quadrant pattern and replicating it to other non-empty quadrants.
    
    1. Analyze the top-left quadrant (0,0 to 13,13) to identify colored regions
    2. Create a new output grid and copy the top-left quadrant
    3. For each other quadrant, check if it's non-empty
    4. If a quadrant is non-empty, replicate the top-left pattern to it
    5. Ensure the central cross (rows and columns 14 and 15) remains black (0)
    6. Return the resulting transformed grid
    
    Returns a new 30x30 ColoredGrid with the transformed pattern.
    """
    # Analyze top-left quadrant
    regions = analyze_top_left(input_grid)
    
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(30)] for _ in range(30)])
    
    # Copy top-left quadrant
    replicate_pattern(input_grid, output, regions, 0, 0, 0, 0)
    
    # Define quadrants
    quadrants = [
        ((0, 16), (13, 29)),  # Top-right
        ((16, 0), (29, 13)),  # Bottom-left
        ((16, 16), (29, 29))  # Bottom-right
    ]
    
    # Process other quadrants
    for (top, left), (bottom, right) in quadrants:
        if is_quadrant_non_empty(input_grid, top, left, bottom, right):
            replicate_pattern(input_grid, output, regions, 0, 0, top, left)
    
    # Ensure central cross remains black
    for i in range(30):
        output.values[14][i] = 0
        output.values[15][i] = 0
        output.values[i][14] = 0
        output.values[i][15] = 0
    
    return output
