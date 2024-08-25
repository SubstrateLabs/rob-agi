from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_15113be4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by introducing or modifying a secondary color (sky blue, magenta, or green)
    in a balanced pattern, primarily in the upper half of the grid. The function follows these steps:
    1. Identifies the secondary color to use (8: sky blue, 6: magenta, or 3: green).
    2. Analyzes existing patterns and identifies potential transformation areas.
    3. Plans and applies L-shaped transformations, focusing on the upper sections.
    4. Balances the design while preserving the yellow grid structure and most of the lower sections.
    5. Makes final adjustments to ensure a visually appealing and intentional pattern.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the applied pattern.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Identify the secondary color
    secondary_color = identify_secondary_color(output_grid)

    # Analyze existing patterns and identify potential areas for transformation
    existing_patterns = find_existing_patterns(output_grid, secondary_color)
    potential_areas = identify_potential_areas(output_grid)

    # Plan and apply L-shaped transformations
    apply_l_shape_transformations(output_grid, potential_areas, secondary_color, existing_patterns)

    # Balance the design and make final adjustments
    balance_design(output_grid, secondary_color)

    return output_grid

def identify_secondary_color(grid: ColoredGrid) -> int:
    colors = grid.get_unique_colors()
    if 8 in colors:
        return 8  # sky blue
    elif 6 in colors:
        return 6  # magenta
    elif 3 in colors:
        return 3  # green
    else:
        return 8  # default to sky blue if no secondary color is present

def find_existing_patterns(grid: ColoredGrid, color: int) -> List[Tuple[int, int, int, int]]:
    return [rect for rect in grid.detect_rectangles() if rect[0] == color]

def identify_potential_areas(grid: ColoredGrid) -> List[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    upper_half = rows // 2
    return [(r, c) for r in range(upper_half) for c in range(cols)
            if grid.get_cell(r, c) == 1 and is_potential_l_shape(grid, r, c)]

def is_potential_l_shape(grid: ColoredGrid, r: int, c: int) -> bool:
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    blue_neighbors = sum(1 for dr, dc in directions
                         if grid.get_cell(r + dr, c + dc) == 1)
    return blue_neighbors >= 2

def apply_l_shape_transformations(grid: ColoredGrid, areas: List[Tuple[int, int]], color: int, existing_patterns: List[Tuple[int, int, int, int]]):
    changes = 0
    for r, c in areas:
        if changes >= 3 or r >= grid.get_dimensions()[0] // 2:
            break
        if create_l_shape(grid, r, c, color):
            changes += 1

def create_l_shape(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    directions = [(0, 0), (0, 1), (1, 0)]
    if all(grid.get_cell(r + dr, c + dc) != 4 for dr, dc in directions):
        for dr, dc in directions:
            grid.set_cell(r + dr, c + dc, color)
        return True
    return False

def balance_design(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows // 2, rows * 3 // 4):
        for c in range(cols):
            if grid.get_cell(r, c) == 1 and random.random() < 0.1:
                grid.set_cell(r, c, color)
