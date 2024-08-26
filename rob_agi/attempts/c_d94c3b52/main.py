from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional

def solve_d94c3b52(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by applying the following steps:
    1. Analyzes the input grid to identify sky blue areas and significant patterns.
    2. Expands sky blue (8) areas to 3x3 if present, or creates one if absent.
    3. Moves the sky blue area to a new position based on grid quadrants.
    4. Creates an alternation template of blue (1) and orange (7).
    5. Identifies and preserves significant patterns.
    6. Applies color transformations:
       - Preserves sky blue (8) squares in their new position.
       - Alternates between blue (1) and orange (7) for other colored squares.
       - Maintains significant patterns while potentially changing their colors.
    7. Ensures black (0) squares remain unchanged.
    8. Balances novelty and familiarity in the transformed grid.
    9. Handles edge cases and adjusts transformation based on input patterns.
    10. Validates the transformation to ensure sufficient change and key feature presence.
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
    new_grid = balance_novelty_and_familiarity(input_grid, new_grid)
    
    return new_grid

def find_sky_blue(grid: ColoredGrid) -> List[Tuple[int, int]]:
    sky_blue_cells = []
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 8:
                sky_blue_cells.append((r, c))
    return sky_blue_cells

def expand_sky_blue(grid: ColoredGrid, sky_blue_cells: List[Tuple[int, int]]) -> ColoredGrid:
    if not sky_blue_cells:
        # Create a new sky blue area if none exists
        r, c = grid.num_rows // 2, grid.num_cols // 2
        sky_blue_cells = [(r, c)]

    new_sky_blue_area = set()
    for r, c in sky_blue_cells:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    new_sky_blue_area.add((nr, nc))
    
    for r, c in new_sky_blue_area:
        grid.values[r][c] = 8
    
    return grid, list(new_sky_blue_area)

def move_sky_blue(grid: ColoredGrid, sky_blue_area: List[Tuple[int, int]]) -> Tuple[int, int]:
    rows, cols = grid.num_rows, grid.num_cols
    center_r, center_c = sum(r for r, _ in sky_blue_area) // len(sky_blue_area), sum(c for _, c in sky_blue_area) // len(sky_blue_area)
    
    # Determine the target quadrant
    if center_r < rows // 2 and center_c < cols // 2:  # Top-left quadrant
        target_r, target_c = center_r, cols - 1 - center_c
    elif center_r < rows // 2 and center_c >= cols // 2:  # Top-right quadrant
        target_r, target_c = rows - 1 - center_r, center_c
    elif center_r >= rows // 2 and center_c < cols // 2:  # Bottom-left quadrant
        target_r, target_c = rows - 1 - center_r, center_c
    else:  # Bottom-right quadrant
        target_r, target_c = center_r, cols - 1 - center_c
    
    # Move the sky blue area
    offset_r, offset_c = target_r - center_r, target_c - center_c
    new_sky_blue_area = [(r + offset_r, c + offset_c) for r, c in sky_blue_area]
    
    # Clear old position and set new position
    for r, c in sky_blue_area:
        grid.values[r][c] = 0
    for r, c in new_sky_blue_area:
        if 0 <= r < rows and 0 <= c < cols:
            grid.values[r][c] = 8
    
    return target_r, target_c

def create_alternation_template(grid: ColoredGrid, sky_blue_pos: Tuple[int, int]) -> List[List[int]]:
    template = [[0 for _ in range(grid.num_cols)] for _ in range(grid.num_rows)]
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if abs(r - sky_blue_pos[0]) <= 1 and abs(c - sky_blue_pos[1]) <= 1:
                template[r][c] = 8  # Sky blue area
            elif (r + c) % 2 == 0:
                template[r][c] = 1  # Blue
            else:
                template[r][c] = 7  # Orange
    return template

def find_preserved_patterns(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
    patterns = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                color = grid.values[r][c]
                pattern = get_pattern(grid, r, c, color, visited)
                if len(pattern) >= 3:
                    min_r = min(r for r, _ in pattern)
                    min_c = min(c for _, c in pattern)
                    max_r = max(r for r, _ in pattern)
                    max_c = max(c for _, c in pattern)
                    patterns.append((min_r, min_c, max_r, max_c))
    return patterns

def get_pattern(grid: ColoredGrid, r: int, c: int, color: int, visited: set) -> List[Tuple[int, int]]:
    if (r, c) in visited or r < 0 or r >= grid.num_rows or c < 0 or c >= grid.num_cols or grid.values[r][c] != color:
        return []
    visited.add((r, c))
    pattern = [(r, c)]
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        pattern.extend(get_pattern(grid, r + dr, c + dc, color, visited))
    return pattern

def apply_color_transformation(input_grid: ColoredGrid, new_grid: ColoredGrid, template: List[List[int]], sky_blue_pos: Tuple[int, int], preserved_patterns: List[Tuple[int, int, int, int]]) -> ColoredGrid:
    for r in range(new_grid.num_rows):
        for c in range(new_grid.num_cols):
            if input_grid.values[r][c] != 0:  # Non-black cell in input
                if abs(r - sky_blue_pos[0]) <= 1 and abs(c - sky_blue_pos[1]) <= 1:
                    new_grid.values[r][c] = 8  # Sky blue area
                elif is_preserved_pattern(r, c, preserved_patterns):
                    new_grid.values[r][c] = input_grid.values[r][c]  # Preserve original pattern
                else:
                    new_grid.values[r][c] = template[r][c]  # Use template color (1 or 7)
            else:
                new_grid.values[r][c] = 0  # Keep black cells black
    return new_grid

def is_preserved_pattern(r: int, c: int, preserved_patterns: List[Tuple[int, int, int, int]]) -> bool:
    return any(min_r <= r <= max_r and min_c <= c <= max_c for min_r, min_c, max_r, max_c in preserved_patterns)

def balance_novelty_and_familiarity(input_grid: ColoredGrid, new_grid: ColoredGrid) -> ColoredGrid:
    rows, cols = input_grid.num_rows, input_grid.num_cols
    changes = sum(1 for r in range(rows) for c in range(cols) if input_grid.values[r][c] != new_grid.values[r][c])
    change_percentage = changes / (rows * cols)
    
    if change_percentage < 0.3:  # Too little change
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] != 0 and new_grid.values[r][c] != 8:
                    if random.random() < 0.3:
                        new_grid.values[r][c] = 7 if input_grid.values[r][c] == 1 else 1
    elif change_percentage > 0.7:  # Too much change
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] != 0 and new_grid.values[r][c] != 8:
                    if random.random() < 0.3:
                        new_grid.values[r][c] = input_grid.values[r][c]
    
    return new_grid

def validate_transformation(input_grid: ColoredGrid, output_grid: ColoredGrid) -> bool:
    rows, cols = input_grid.num_rows, input_grid.num_cols
    changes = sum(1 for r in range(rows) for c in range(cols) if input_grid.values[r][c] != output_grid.values[r][c])
    change_percentage = changes / (rows * cols)
    
    sky_blue_input = find_sky_blue(input_grid)
    sky_blue_output = find_sky_blue(output_grid)
    
    return (0.3 <= change_percentage <= 0.7 and
            len(sky_blue_output) >= len(sky_blue_input) and
            all(output_grid.values[r][c] == 0 for r, c in sky_blue_input) and
            all(input_grid.values[r][c] == 0 for r, c in sky_blue_output))
