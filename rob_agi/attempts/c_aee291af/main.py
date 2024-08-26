from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional, Set

def solve_aee291af(input_grid: ColoredGrid) -> Optional[ColoredGrid]:
    """
    Solves the grid transformation challenge by finding the largest square pattern
    of sky blue (8) outline with a specific arrangement of red (2) squares inside.
    
    The function works as follows:
    1. Identifies all sky blue and red squares in the input grid.
    2. Searches for valid 5x5 patterns first, then 4x4 patterns.
    3. Returns the largest valid pattern found, or None if no valid pattern exists.
    
    Valid patterns:
    - 5x5 grid: Sky blue outline with a red cross or X pattern in the center
    - 4x4 grid: Sky blue outline with a 2x2 red square in any of the 4 positions inside
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    Optional[ColoredGrid]: The transformed grid containing the largest valid pattern, or None if no pattern is found.
    """
    if input_grid.num_rows < 4 or input_grid.num_cols < 4:
        return None  # Grid is too small to contain any valid pattern

    sky_blue_coords = set(find_color_coords(input_grid, 8))
    red_coords = set(find_color_coords(input_grid, 2))
    
    for r in range(input_grid.num_rows - 4):
        for c in range(input_grid.num_cols - 4):
            solution_5x5 = check_5x5_pattern((r, c), sky_blue_coords, red_coords)
            if solution_5x5:
                return solution_5x5
    
    for r in range(input_grid.num_rows - 3):
        for c in range(input_grid.num_cols - 3):
            solution_4x4 = check_4x4_pattern((r, c), sky_blue_coords, red_coords)
            if solution_4x4:
                return solution_4x4
    
    return None

def find_color_coords(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == color]

def check_5x5_pattern(top_left: Tuple[int, int], sky_blue_coords: Set[Tuple[int, int]], red_coords: Set[Tuple[int, int]]) -> Optional[ColoredGrid]:
    r, c = top_left
    outline = {(r+i, c+j) for i in range(5) for j in range(5) if i in {0, 4} or j in {0, 4}}
    
    if not outline.issubset(sky_blue_coords):
        return None
    
    cross_pattern = {(r+1, c+2), (r+2, c+1), (r+2, c+2), (r+2, c+3), (r+3, c+2)}
    x_pattern = {(r+1, c+1), (r+1, c+3), (r+2, c+2), (r+3, c+1), (r+3, c+3)}
    
    if cross_pattern.issubset(red_coords) or x_pattern.issubset(red_coords):
        solution = [[8 for _ in range(5)] for _ in range(5)]
        for rr, cc in (cross_pattern if cross_pattern.issubset(red_coords) else x_pattern):
            solution[rr-r][cc-c] = 2
        return ColoredGrid(values=solution)
    return None

def check_4x4_pattern(top_left: Tuple[int, int], sky_blue_coords: Set[Tuple[int, int]], red_coords: Set[Tuple[int, int]]) -> Optional[ColoredGrid]:
    r, c = top_left
    outline = {(r+i, c+j) for i in range(4) for j in range(4) if i in {0, 3} or j in {0, 3}}
    
    if not outline.issubset(sky_blue_coords):
        return None
    
    for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        red_square = {(r+1+dr, c+1+dc), (r+1+dr, c+2+dc), (r+2+dr, c+1+dc), (r+2+dr, c+2+dc)}
        if red_square.issubset(red_coords):
            solution = [[8 for _ in range(4)] for _ in range(4)]
            for rr, cc in red_square:
                solution[rr-r][cc-c] = 2
            return ColoredGrid(values=solution)
    return None
