from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d94c3b52(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Identifies and expands sky blue (8) squares to 3x3 if present.
    2. Creates an alternation template based on the grid structure.
    3. Applies color transformations:
       - Moves sky blue squares to a new position.
       - Alternates between blue (1) and orange (7) for other colored squares.
    4. Maintains the overall structure and symmetry of the input patterns.
    5. Ensures black (0) squares remain unchanged.
    """
    new_grid = input_grid.deep_copy()
    sky_blue_pos = find_sky_blue(new_grid)
    
    if sky_blue_pos:
        new_grid = expand_sky_blue(new_grid, sky_blue_pos)
        new_sky_blue_pos = move_sky_blue(new_grid, sky_blue_pos)
    else:
        new_sky_blue_pos = None
    
    template = create_alternation_template(new_grid, new_sky_blue_pos)
    new_grid = apply_color_transformation(new_grid, template, new_sky_blue_pos)
    
    return new_grid

def find_sky_blue(grid: ColoredGrid) -> Tuple[int, int]:
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 8:
                return r, c
    return None

def expand_sky_blue(grid: ColoredGrid, pos: Tuple[int, int]) -> ColoredGrid:
    r, c = pos
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                grid.values[nr][nc] = 8
    return grid

def move_sky_blue(grid: ColoredGrid, old_pos: Tuple[int, int]) -> Tuple[int, int]:
    r, c = old_pos
    new_r = grid.num_rows - 1 - r if r < grid.num_rows // 2 else r - grid.num_rows // 2
    new_c = grid.num_cols - 1 - c if c < grid.num_cols // 2 else c - grid.num_cols // 2
    
    # Move the 3x3 sky blue square
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            grid.values[new_r + dr][new_c + dc] = 8
            grid.values[r + dr][c + dc] = 0  # Clear the old position
    
    return new_r, new_c

def create_alternation_template(grid: ColoredGrid, sky_blue_pos: Tuple[int, int]) -> List[List[int]]:
    template = [[0 for _ in range(grid.num_cols)] for _ in range(grid.num_rows)]
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if sky_blue_pos and abs(r - sky_blue_pos[0]) <= 1 and abs(c - sky_blue_pos[1]) <= 1:
                template[r][c] = 2  # Sky blue area
            elif (r + c) % 2 == 0:
                template[r][c] = 0  # Blue
            else:
                template[r][c] = 1  # Orange
    return template

def apply_color_transformation(grid: ColoredGrid, template: List[List[int]], sky_blue_pos: Tuple[int, int]) -> ColoredGrid:
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] != 0:  # Non-black cell
                if template[r][c] == 2:  # Sky blue area
                    grid.values[r][c] = 8
                elif template[r][c] == 0:
                    grid.values[r][c] = 1  # Blue
                else:
                    grid.values[r][c] = 7  # Orange
    return grid
