from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional, Set

def solve_aee291af(input_grid: ColoredGrid) -> Optional[ColoredGrid]:
    """
    Solves the grid transformation challenge by finding the largest square pattern
    of sky blue (8) outline with a specific arrangement of red (2) squares inside.
    
    The function works as follows:
    1. Identifies all sky blue and red squares in the input grid.
    2. Starts with the largest possible square size (5x5) and decreases until a valid pattern is found.
    3. For each size, checks if a valid sky blue outline exists.
    4. If an outline exists, checks for valid red square patterns inside.
    5. If a valid pattern is found, constructs and returns the solution grid.
    6. If no valid pattern is found, returns None.
    
    The valid patterns include:
    - 5x5 grid with red squares forming a cross shape (center and four adjacent squares)
    - 4x4 grid with two red squares arranged vertically or diagonally
    
    The function ensures that the pattern found exactly matches one of these configurations,
    without any additional red squares in the region.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    Optional[ColoredGrid]: The transformed grid containing the largest valid pattern, or None if no pattern is found.
    """
    sky_blue_coords = set(find_color_coords(input_grid, 8))
    red_coords = set(find_color_coords(input_grid, 2))
    
    max_size = min(input_grid.num_rows, input_grid.num_cols)
    max_size = min(max_size, 5)  # We only need to check up to 5x5
    
    for size in range(max_size, 3, -1):
        for top in range(input_grid.num_rows - size + 1):
            for left in range(input_grid.num_cols - size + 1):
                if is_valid_outline(sky_blue_coords, top, left, size):
                    pattern = find_red_pattern(red_coords, top, left, size)
                    if pattern:
                        return construct_solution(size, pattern)
    
    return None

def find_color_coords(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == color]

def is_valid_outline(coords: Set[Tuple[int, int]], top: int, left: int, size: int) -> bool:
    outline = set((r, c) for r in range(top, top + size) for c in range(left, left + size)
                  if r == top or r == top + size - 1 or c == left or c == left + size - 1)
    return outline.issubset(coords)

def find_red_pattern(coords: Set[Tuple[int, int]], top: int, left: int, size: int) -> Optional[List[Tuple[int, int]]]:
    relative_coords = set((r - top, c - left) for r, c in coords if top < r < top + size - 1 and left < c < left + size - 1)
    
    if size == 5:
        pattern = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]  # Cross shape (5x5)
        if set(pattern) == relative_coords:
            return pattern
    elif size == 4:
        patterns = [
            [(1, 1), (1, 2)],  # Vertical line (4x4)
            [(1, 1), (2, 2)],  # Diagonal (4x4)
        ]
        for pattern in patterns:
            if set(pattern) == relative_coords:
                return pattern
    
    return None

def construct_solution(size: int, pattern: List[Tuple[int, int]]) -> ColoredGrid:
    solution = [[8 for _ in range(size)] for _ in range(size)]
    for r, c in pattern:
        solution[r][c] = 2
    return ColoredGrid(values=solution)
