from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional, Set

def solve_aee291af(input_grid: ColoredGrid) -> Optional[ColoredGrid]:
    """
    Solves the grid transformation challenge by finding the largest square pattern
    of sky blue (8) outline with a specific arrangement of red (2) squares inside.
    
    The function works as follows:
    1. Identifies all sky blue and red squares in the input grid.
    2. Finds all 2x2 red squares as potential centers for patterns.
    3. For each 2x2 red square, checks for valid 5x5 patterns first, then 4x4 patterns.
    4. Returns the largest valid pattern found, or None if no valid pattern exists.
    
    Valid patterns:
    - 5x5 grid: Sky blue outline with a red cross in the center (5 red squares)
    - 4x4 grid: Sky blue outline with a 2x2 red square in any of the 4 positions inside
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    Optional[ColoredGrid]: The transformed grid containing the largest valid pattern, or None if no pattern is found.
    """
    sky_blue_coords = set(find_color_coords(input_grid, 8))
    red_coords = set(find_color_coords(input_grid, 2))
    
    red_2x2_squares = find_2x2_squares(red_coords)
    
    for center in red_2x2_squares:
        solution_5x5 = check_5x5_pattern(center, sky_blue_coords, red_coords)
        if solution_5x5:
            return solution_5x5
    
    for center in red_2x2_squares:
        solution_4x4 = check_4x4_pattern(center, sky_blue_coords, red_coords)
        if solution_4x4:
            return solution_4x4
    
    return None

def find_color_coords(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == color]

def find_2x2_squares(coords: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    centers = []
    for r, c in coords:
        if (r, c+1) in coords and (r+1, c) in coords and (r+1, c+1) in coords:
            centers.append((r, c))
    return centers

def check_5x5_pattern(center: Tuple[int, int], sky_blue_coords: Set[Tuple[int, int]], red_coords: Set[Tuple[int, int]]) -> Optional[ColoredGrid]:
    r, c = center
    outline = {(r-2, c-2), (r-2, c-1), (r-2, c), (r-2, c+1), (r-2, c+2),
               (r-1, c-2), (r-1, c+2),
               (r, c-2), (r, c+2),
               (r+1, c-2), (r+1, c+2),
               (r+2, c-2), (r+2, c-1), (r+2, c), (r+2, c+1), (r+2, c+2)}
    
    cross = {(r-1, c), (r, c-1), (r, c), (r, c+1), (r+1, c)}
    
    if outline.issubset(sky_blue_coords) and cross.issubset(red_coords):
        solution = [[8 for _ in range(5)] for _ in range(5)]
        for rr, cc in cross:
            solution[rr-r+2][cc-c+2] = 2
        return ColoredGrid(values=solution)
    return None

def check_4x4_pattern(center: Tuple[int, int], sky_blue_coords: Set[Tuple[int, int]], red_coords: Set[Tuple[int, int]]) -> Optional[ColoredGrid]:
    r, c = center
    outline = {(r-1, c-1), (r-1, c), (r-1, c+1), (r-1, c+2),
               (r, c-1), (r, c+2),
               (r+1, c-1), (r+1, c+2),
               (r+2, c-1), (r+2, c), (r+2, c+1), (r+2, c+2)}
    
    if outline.issubset(sky_blue_coords):
        for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            red_square = {(r+dr, c+dc), (r+dr, c+dc+1), (r+dr+1, c+dc), (r+dr+1, c+dc+1)}
            if red_square.issubset(red_coords):
                solution = [[8 for _ in range(4)] for _ in range(4)]
                for rr, cc in red_square:
                    solution[rr-r+1][cc-c+1] = 2
                return ColoredGrid(values=solution)
    return None
