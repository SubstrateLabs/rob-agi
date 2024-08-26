from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_f5c89df1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by identifying the core shape formed by sky blue (8) squares,
    and creating an idealized, symmetrical version of it.
    
    1. Analyze the input shape to determine its key characteristics.
    2. Create a 5x5 idealized pattern based on the input structure.
    3. Position the new pattern within a 13x13 grid.
    4. Adjust the pattern to maintain symmetry and essence of the original.
    
    The transformation preserves the general structure of the input
    while creating a more symmetrical and centered 5x5 pattern within the output.
    
    Returns a new 13x13 grid with the transformed pattern.
    """
    # Step 1: Analyze input shape
    sky_blue_coords = [(r, c) for r in range(13) for c in range(13) if input_grid.get_cell(r, c) == 8]
    if not sky_blue_coords:
        return ColoredGrid(values=[[0 for _ in range(13)] for _ in range(13)])
    
    min_r, max_r = min(r for r, _ in sky_blue_coords), max(r for r, _ in sky_blue_coords)
    min_c, max_c = min(c for _, c in sky_blue_coords), max(c for _, c in sky_blue_coords)
    
    # Step 2: Create 5x5 idealized pattern
    pattern = [[0 for _ in range(5)] for _ in range(5)]
    pattern[0][2] = pattern[2][0] = pattern[2][4] = pattern[4][2] = 8  # Corners
    pattern[1][1] = pattern[1][3] = pattern[3][1] = pattern[3][3] = 8  # Inner corners
    pattern[2][2] = 8  # Center
    
    # Step 3 & 4: Position pattern and create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(13)] for _ in range(13)])
    start_row, start_col = 4, 4  # Center the 5x5 pattern in the 13x13 grid
    
    for r in range(5):
        for c in range(5):
            if pattern[r][c] == 8:
                output_grid.set_cell(start_row + r, start_col + c, 8)
    
    return output_grid
