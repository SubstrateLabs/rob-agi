from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bb52a14b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the bb52a14b challenge by finding a distinctive color pattern
    in the left half of the grid and replicating it in suitable areas
    on the right side of the grid.

    1. Find the largest non-black color pattern in the left half.
    2. Replicate this pattern in the center-right area.
    3. If suitable, replicate in the top-right and bottom-right areas.
    4. Maintain original scattered colors outside the replicated areas.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with replicated patterns.
    """
    # Step 1: Analyze the input grid and find the pattern
    pattern = find_largest_pattern(input_grid)
    
    # Step 2: Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Step 3: Replicate in center-right
    replicate_pattern(output_grid, pattern, 11, 17)
    
    # Step 4: Replicate in top-right if suitable
    if is_area_suitable(output_grid, 1, 19, len(pattern), len(pattern[0])):
        replicate_pattern(output_grid, pattern, 1, 19)
    
    # Step 5: Replicate in bottom-right if suitable
    if is_area_suitable(output_grid, 18, 17, len(pattern), len(pattern[0])):
        replicate_pattern(output_grid, pattern, 18, 17)
    
    return output_grid

def find_largest_pattern(grid: ColoredGrid) -> List[List[int]]:
    """Find the largest non-black color pattern in the left half of the grid."""
    rows, cols = grid.get_dimensions()
    largest_pattern = []
    for r in range(rows):
        for c in range(cols // 2):
            if grid.get_cell(r, c) != 0:
                pattern = extract_pattern(grid, r, c)
                if len(pattern) * len(pattern[0]) > len(largest_pattern) * len(largest_pattern[0] if largest_pattern else []):
                    largest_pattern = pattern
    return largest_pattern

def extract_pattern(grid: ColoredGrid, start_r: int, start_c: int) -> List[List[int]]:
    """Extract a contiguous non-black color pattern starting from the given position."""
    rows, cols = grid.get_dimensions()
    pattern = []
    r, c = start_r, start_c
    while r < rows and grid.get_cell(r, c) != 0:
        row = []
        while c < cols // 2 and grid.get_cell(r, c) != 0:
            row.append(grid.get_cell(r, c))
            c += 1
        pattern.append(row)
        r += 1
        c = start_c
    return pattern

def is_area_suitable(grid: ColoredGrid, start_r: int, start_c: int, height: int, width: int) -> bool:
    """Check if the area is suitable for pattern replication (mostly black)."""
    rows, cols = grid.get_dimensions()
    black_count = sum(1 for r in range(start_r, min(start_r + height, rows))
                      for c in range(start_c, min(start_c + width, cols))
                      if grid.get_cell(r, c) == 0)
    total_cells = height * width
    return black_count / total_cells > 0.7  # 70% black threshold

def replicate_pattern(grid: ColoredGrid, pattern: List[List[int]], start_r: int, start_c: int):
    """Replicate the given pattern at the specified position in the grid."""
    rows, cols = grid.get_dimensions()
    for r, row in enumerate(pattern):
        for c, value in enumerate(row):
            if start_r + r < rows and start_c + c < cols:
                grid.set_cell(start_r + r, start_c + c, value)
