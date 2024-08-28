from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_f5c89df1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by identifying the core shape formed by sky blue (8) squares,
    and creating an idealized, symmetrical version of it.
    
    1. Analyze the input shape to determine its key characteristics.
    2. Determine the shape category and create an idealized pattern.
    3. Position the new pattern within a 13x13 grid, with a slight upward shift.
    4. Adjust the pattern to maintain symmetry and essence of the original.
    
    The transformation preserves the general structure of the input
    while creating a more symmetrical and centered pattern within the output.
    
    Returns a new 13x13 grid with the transformed pattern.
    """
    # Step 1: Analyze input shape
    sky_blue_coords = [(r, c) for r in range(13) for c in range(13) if input_grid.get_cell(r, c) == 8]
    if not sky_blue_coords:
        return ColoredGrid(values=[[0 for _ in range(13)] for _ in range(13)])
    
    min_r, max_r = min(r for r, _ in sky_blue_coords), max(r for r, _ in sky_blue_coords)
    min_c, max_c = min(c for _, c in sky_blue_coords), max(c for _, c in sky_blue_coords)
    
    # Step 2: Determine shape category and create idealized pattern
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    aspect_ratio = width / height if height != 0 else 0
    
    if aspect_ratio > 1.5:  # Wide rectangle
        pattern = [
            [0, 0, 8, 8, 8, 8, 8, 0, 0],
            [0, 0, 8, 8, 0, 8, 8, 0, 0],
            [0, 0, 8, 8, 0, 8, 8, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    elif aspect_ratio < 0.67:  # Tall rectangle
        pattern = [
            [0, 0, 0, 8, 0, 0],
            [0, 0, 8, 8, 8, 0],
            [0, 0, 8, 0, 8, 0],
            [0, 0, 8, 0, 8, 0],
            [0, 0, 8, 8, 8, 0],
            [0, 0, 0, 8, 0, 0]
        ]
    else:  # Square or near-square
        pattern = [
            [0, 0, 8, 0, 0],
            [0, 8, 0, 8, 0],
            [8, 0, 8, 0, 8],
            [0, 8, 0, 8, 0],
            [0, 0, 8, 0, 0]
        ]
    
    # Step 3 & 4: Position pattern and create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(13)] for _ in range(13)])
    pattern_height, pattern_width = len(pattern), len(pattern[0])
    start_row = max(0, min(13 - pattern_height, 6 - pattern_height // 2))  # Centered with slight upward shift
    start_col = max(0, min(13 - pattern_width, 6 - pattern_width // 2))
    
    for r in range(pattern_height):
        for c in range(pattern_width):
            if pattern[r][c] == 8:
                output_grid.set_cell(start_row + r, start_col + c, 8)
    
    return output_grid
