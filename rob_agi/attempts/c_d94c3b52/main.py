from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_d94c3b52(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Identifies and expands sky blue (8) squares to 3x3 if present.
    2. Moves the sky blue area to a new position based on grid structure.
    3. Creates an alternation template based on the grid structure.
    4. Applies color transformations:
       - Preserves sky blue (8) squares in their new position.
       - Alternates between blue (1) and orange (7) for other colored squares.
    5. Maintains the overall structure and patterns of the input grid.
    6. Ensures black (0) squares remain unchanged.
    7. Preserves original patterns in their positions when not conflicting with sky blue area.
    8. Handles edge cases and adjusts transformation based on input patterns.
    """
    new_grid = input_grid.deep_copy()
    sky_blue_pos = find_sky_blue(new_grid)
    
    if sky_blue_pos:
        new_grid = expand_sky_blue(new_grid, sky_blue_pos)
        new_sky_blue_pos = move_sky_blue(new_grid, sky_blue_pos)
    else:
        new_sky_blue_pos = None
    
    template = create_alternation_template(new_grid, new_sky_blue_pos)
    preserved_patterns = find_preserved_patterns(input_grid)
    new_grid = apply_color_transformation(input_grid, new_grid, template, new_sky_blue_pos, preserved_patterns)
    
    return new_grid

def find_sky_blue(grid: ColoredGrid) -> Optional[Tuple[int, int]]:
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
    rows, cols = grid.num_rows, grid.num_cols
    
    # Calculate new position based on grid quadrants
    if r < rows // 2 and c < cols // 2:  # Top-left quadrant
        new_r, new_c = r, cols - 1 - c
    elif r < rows // 2 and c >= cols // 2:  # Top-right quadrant
        new_r, new_c = rows - 1 - r, c
    elif r >= rows // 2 and c < cols // 2:  # Bottom-left quadrant
        new_r, new_c = r, cols - 1 - c
    else:  # Bottom-right quadrant
        new_r, new_c = rows - 1 - r, c
    
    # Move the 3x3 sky blue square
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            grid.values[new_r + dr][new_c + dc] = 8
            if (r + dr, c + dc) != (new_r + dr, new_c + dc):
                grid.values[r + dr][c + dc] = 0  # Clear the old position
    
    return new_r, new_c

def create_alternation_template(grid: ColoredGrid, sky_blue_pos: Optional[Tuple[int, int]]) -> List[List[int]]:
    template = [[0 for _ in range(grid.num_cols)] for _ in range(grid.num_rows)]
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if sky_blue_pos and abs(r - sky_blue_pos[0]) <= 1 and abs(c - sky_blue_pos[1]) <= 1:
                template[r][c] = 8  # Sky blue area
            elif (r + c) % 2 == 0:
                template[r][c] = 1  # Blue
            else:
                template[r][c] = 7  # Orange
    return template

def find_preserved_patterns(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    patterns = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                color = grid.values[r][c]
                size = get_pattern_size(grid, r, c, color, visited)
                if size >= 3:
                    patterns.append((r, c, color))
    return patterns

def get_pattern_size(grid: ColoredGrid, r: int, c: int, color: int, visited: set) -> int:
    if (r, c) in visited or r < 0 or r >= grid.num_rows or c < 0 or c >= grid.num_cols or grid.values[r][c] != color:
        return 0
    visited.add((r, c))
    size = 1
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        size += get_pattern_size(grid, r + dr, c + dc, color, visited)
    return size

def apply_color_transformation(input_grid: ColoredGrid, new_grid: ColoredGrid, template: List[List[int]], sky_blue_pos: Optional[Tuple[int, int]], preserved_patterns: List[Tuple[int, int, int]]) -> ColoredGrid:
    for r in range(new_grid.num_rows):
        for c in range(new_grid.num_cols):
            if input_grid.values[r][c] != 0:  # Non-black cell in input
                if sky_blue_pos and abs(r - sky_blue_pos[0]) <= 1 and abs(c - sky_blue_pos[1]) <= 1:
                    new_grid.values[r][c] = 8  # Sky blue area
                elif template[r][c] == 8:
                    new_grid.values[r][c] = 8  # Keep sky blue in its new position
                elif is_preserved_pattern(r, c, preserved_patterns):
                    new_grid.values[r][c] = input_grid.values[r][c]  # Preserve original pattern
                else:
                    new_grid.values[r][c] = template[r][c]  # Use template color (1 or 7)
            else:
                new_grid.values[r][c] = 0  # Keep black cells black
    return new_grid

def is_preserved_pattern(r: int, c: int, preserved_patterns: List[Tuple[int, int, int]]) -> bool:
    return any(abs(r - pr) < 3 and abs(c - pc) < 3 for pr, pc, _ in preserved_patterns)
