from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def analyze_quadrant(grid: ColoredGrid, top: int, left: int, bottom: int, right: int) -> List[Tuple[int, int, int, int, int]]:
    """Analyze a quadrant and return a list of colored regions."""
    regions = []
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            color = grid.values[i][j]
            if color != 0:
                regions.append((color, i - top, j - left, i - top, j - left))
    
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

def flip_pattern(pattern: List[Tuple[int, int, int, int, int]], horizontal: bool, vertical: bool, size: int) -> List[Tuple[int, int, int, int, int]]:
    """Flip a pattern horizontally and/or vertically."""
    flipped = []
    for color, top, left, bottom, right in pattern:
        if horizontal:
            left, right = size - 1 - right, size - 1 - left
        if vertical:
            top, bottom = size - 1 - bottom, size - 1 - top
        flipped.append((color, top, left, bottom, right))
    return flipped

def apply_pattern(grid: ColoredGrid, pattern: List[Tuple[int, int, int, int, int]], top: int, left: int):
    """Apply a pattern to a specific position in the grid."""
    for color, r_top, r_left, r_bottom, r_right in pattern:
        for i in range(r_top, r_bottom + 1):
            for j in range(r_left, r_right + 1):
                grid.values[top + i][left + j] = color

def solve_40f6cd08(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by analyzing patterns in non-empty quadrants, transforming them, and applying them to create a symmetric output.
    
    1. Analyze each quadrant to identify colored regions
    2. Determine the base pattern from the non-empty quadrant(s)
    3. Create transformed patterns for each quadrant using horizontal and vertical flips
    4. Apply the transformed patterns to their respective quadrants in the output grid
    5. Ensure the central cross (rows and columns 14 and 15) remains black (0)
    6. Return the resulting transformed grid
    
    Returns a new 30x30 ColoredGrid with the transformed pattern.
    """
    # Define quadrants
    quadrants = [
        ((0, 0), (13, 13)),    # Top-left
        ((0, 16), (13, 29)),   # Top-right
        ((16, 0), (29, 13)),   # Bottom-left
        ((16, 16), (29, 29))   # Bottom-right
    ]
    
    # Analyze quadrants and find the base pattern
    base_pattern = None
    for (top, left), (bottom, right) in quadrants:
        if is_quadrant_non_empty(input_grid, top, left, bottom, right):
            base_pattern = analyze_quadrant(input_grid, top, left, bottom, right)
            break
    
    if base_pattern is None:
        return input_grid  # If all quadrants are empty, return the input grid
    
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(30)] for _ in range(30)])
    
    # Apply transformed patterns to each quadrant
    quadrant_size = 14
    transformations = [
        (False, False),  # Top-left: no transformation
        (True, False),   # Top-right: horizontal flip
        (False, True),   # Bottom-left: vertical flip
        (True, True)     # Bottom-right: both flips
    ]
    
    for ((top, left), _), (flip_h, flip_v) in zip(quadrants, transformations):
        pattern = flip_pattern(base_pattern, flip_h, flip_v, quadrant_size)
        apply_pattern(output, pattern, top, left)
    
    # Ensure central cross remains black
    for i in range(30):
        output.values[14][i] = 0
        output.values[15][i] = 0
        output.values[i][14] = 0
        output.values[i][15] = 0
    
    return output
