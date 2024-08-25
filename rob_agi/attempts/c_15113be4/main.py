from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_15113be4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by enhancing or introducing a secondary color (sky blue, magenta, or green)
    in a balanced pattern, focusing on the top-left quadrant. The function follows these steps:
    1. Identifies the secondary color to use (3: green, 6: magenta, or 8: sky blue).
    2. Enhances existing secondary color areas by forming L-shapes in the top-left quadrant.
    3. Introduces new L-shapes in the top-left quadrant in a systematic pattern.
    4. Preserves the yellow grid structure throughout the process.
    5. Makes minimal changes to other quadrants.
    6. Performs a final pass for consistency.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the applied pattern.
    """
    output_grid = input_grid.deep_copy()
    secondary_color = identify_secondary_color(output_grid)

    enhance_top_left_quadrant(output_grid, secondary_color)
    introduce_new_l_shapes(output_grid, secondary_color)
    minimal_changes_other_quadrants(output_grid, secondary_color)
    final_consistency_pass(output_grid, secondary_color)

    return output_grid

def identify_secondary_color(grid: ColoredGrid) -> int:
    colors = grid.get_unique_colors()
    if 3 in colors:
        return 3  # green
    elif 6 in colors:
        return 6  # magenta
    elif 8 in colors:
        return 8  # sky blue
    else:
        return 8  # default to sky blue if no secondary color is present

def enhance_top_left_quadrant(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows // 2):
        for c in range(cols // 2):
            if grid.get_cell(r, c) == color:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [0, 1]:
                        if forms_l_shape(grid, r, c, nr, nc, color):
                            grid.set_cell(nr, nc, color)

def introduce_new_l_shapes(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(0, rows // 2, 3):
        for c in range(0, cols // 2, 3):
            if not any(grid.get_cell(r+dr, c+dc) == color for dr in range(3) for dc in range(3)):
                if is_valid_cell(grid, r, c) and is_valid_cell(grid, r+2, c+2):
                    grid.set_cell(r + (2 if (r // 3 + c // 3) % 2 == 0 else 0),
                                  c + (2 if (r // 3 + c // 3) % 2 == 0 else 0), color)
                    grid.set_cell(r + (1 if (r // 3 + c // 3) % 2 == 0 else 0),
                                  c + (0 if (r // 3 + c // 3) % 2 == 0 else 1), color)

def minimal_changes_other_quadrants(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows // 2, rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color and not has_adjacent_color(grid, r, c, color):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [0, 1]:
                        grid.set_cell(nr, nc, color)
                        break

def final_consistency_pass(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color and not has_adjacent_color(grid, r, c, color):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [0, 1]:
                        grid.set_cell(nr, nc, color)
                        break

def is_valid_cell(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    return 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) != 4

def forms_l_shape(grid: ColoredGrid, r1: int, c1: int, r2: int, c2: int, color: int) -> bool:
    if r1 == r2:
        return (is_valid_cell(grid, r1-1, c1) and grid.get_cell(r1-1, c1) == color) or \
               (is_valid_cell(grid, r1+1, c1) and grid.get_cell(r1+1, c1) == color)
    else:
        return (is_valid_cell(grid, r1, c1-1) and grid.get_cell(r1, c1-1) == color) or \
               (is_valid_cell(grid, r1, c1+1) and grid.get_cell(r1, c1+1) == color)

def has_adjacent_color(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) == color:
            return True
    return False
