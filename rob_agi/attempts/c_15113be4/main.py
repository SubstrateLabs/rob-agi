from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_15113be4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by enhancing or introducing a secondary color (green, magenta, or sky blue)
    in a balanced and visually interesting pattern across all quadrants. The function follows these steps:
    1. Identifies the secondary color to use (3: green, 6: magenta, or 8: sky blue).
    2. Preserves existing patterns of the secondary color.
    3. Enhances existing patterns by forming L-shapes around fixed secondary color cells.
    4. Creates new L-shapes, especially near blue (1) dots.
    5. Fills isolated secondary color dots by extending them into L-shapes.
    6. Balances the distribution of the secondary color across quadrants.
    7. Refines the transformation by addressing any remaining isolated secondary color dots.
    8. Preserves the yellow (4) grid structure throughout the process.
    9. Adds final touches for visual interest, such as diagonal paths or clusters.

    The transformation aims to create a balanced, aesthetically pleasing distribution of L-shapes
    of the secondary color while maintaining the original grid's structure and enhancing existing patterns.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the applied pattern.
    """
    output_grid = input_grid.deep_copy()
    secondary_color = identify_secondary_color(output_grid)
    
    preserve_existing_patterns(output_grid, secondary_color)
    enhance_existing_patterns(output_grid, secondary_color)
    create_new_l_shapes(output_grid, secondary_color)
    fill_isolated_dots(output_grid, secondary_color)
    balance_distribution(output_grid, secondary_color)
    refine_transformation(output_grid, secondary_color)
    add_final_touches(output_grid, secondary_color)

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
        return 3  # default to green if no secondary color is present

def preserve_existing_patterns(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color:
                grid.set_cell(r, c, -color)  # Mark as fixed

def enhance_existing_patterns(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == -color:  # Fixed secondary color cell
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [0, 1]:
                        if forms_l_shape(grid, r, c, -color, nr, nc):
                            grid.set_cell(nr, nc, color)
    
    # Unmark fixed cells
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == -color:
                grid.set_cell(r, c, color)

def create_new_l_shapes(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1 and not has_adjacent_color(grid, r, c, color):
                create_l_shape(grid, r, c, color)

def fill_isolated_dots(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color and not has_adjacent_color(grid, r, c, color):
                extend_to_l_shape(grid, r, c, color)

def balance_distribution(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    quadrants = [
        (0, rows//2, 0, cols//2),
        (0, rows//2, cols//2, cols),
        (rows//2, rows, 0, cols//2),
        (rows//2, rows, cols//2, cols)
    ]
    counts = [count_color_in_quadrant(grid, color, *q) for q in quadrants]
    avg_count = sum(counts) / len(counts)
    
    for i, (r_start, r_end, c_start, c_end) in enumerate(quadrants):
        diff = int(avg_count - counts[i])
        if diff > 0:
            add_l_shapes_to_quadrant(grid, color, r_start, r_end, c_start, c_end, diff)

def refine_transformation(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color and not has_adjacent_color(grid, r, c, color):
                if not extend_to_l_shape(grid, r, c, color):
                    grid.set_cell(r, c, 0)  # Remove if can't extend

def add_final_touches(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.get_cell(r+dr, c+dc) in [0, 1] for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]):
                grid.set_cell(r, c, color)
                grid.set_cell(r+1, c+1, color)

def is_valid_cell(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    return 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) != 4

def has_adjacent_color(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [color, -color]:
            return True
    return False

def forms_l_shape(grid: ColoredGrid, r1: int, c1: int, color: int, r2: int, c2: int) -> bool:
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r2 + dr, c2 + dc
        if (nr, nc) != (r1, c1) and is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [color, -color]:
            return True
    return False

def create_l_shape(grid: ColoredGrid, r: int, c: int, color: int):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dr1, dc1) in enumerate(directions):
        for dr2, dc2 in directions[i+1:]:
            if (is_valid_cell(grid, r+dr1, c+dc1) and grid.get_cell(r+dr1, c+dc1) in [0, 1] and
                is_valid_cell(grid, r+dr2, c+dc2) and grid.get_cell(r+dr2, c+dc2) in [0, 1]):
                grid.set_cell(r, c, color)
                grid.set_cell(r+dr1, c+dc1, color)
                grid.set_cell(r+dr2, c+dc2, color)
                return True
    return False

def extend_to_l_shape(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr1, dc1 in directions:
        for dr2, dc2 in directions:
            if (dr1, dc1) != (dr2, dc2):
                nr1, nc1 = r + dr1, c + dc1
                nr2, nc2 = r + dr2, c + dc2
                if (is_valid_cell(grid, nr1, nc1) and grid.get_cell(nr1, nc1) in [0, 1] and
                    is_valid_cell(grid, nr2, nc2) and grid.get_cell(nr2, nc2) in [0, 1]):
                    grid.set_cell(nr1, nc1, color)
                    grid.set_cell(nr2, nc2, color)
                    return True
    return False

def count_color_in_quadrant(grid: ColoredGrid, color: int, r_start: int, r_end: int, c_start: int, c_end: int) -> int:
    return sum(1 for r in range(r_start, r_end) for c in range(c_start, c_end) if grid.get_cell(r, c) == color)

def add_l_shapes_to_quadrant(grid: ColoredGrid, color: int, r_start: int, r_end: int, c_start: int, c_end: int, count: int):
    added = 0
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            if added >= count:
                return
            if grid.get_cell(r, c) == 1 and not has_adjacent_color(grid, r, c, color):
                if create_l_shape(grid, r, c, color):
                    added += 1
