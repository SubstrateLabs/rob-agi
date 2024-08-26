from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_15113be4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by enhancing or introducing a secondary color (sky blue, magenta, or green)
    in a balanced pattern across all quadrants. The function follows these steps:
    1. Identifies the secondary color to use (3: green, 6: magenta, or 8: sky blue).
    2. Analyzes the existing pattern and distribution of the secondary color.
    3. Enhances existing secondary color areas by forming L-shapes or small clusters.
    4. Introduces new instances of the secondary color in a balanced manner.
    5. Preserves the yellow grid structure throughout the process.
    6. Performs a final pass for consistency and balance.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the applied pattern.
    """
    output_grid = input_grid.deep_copy()
    secondary_color = identify_secondary_color(output_grid)
    
    analyze_pattern(output_grid, secondary_color)
    enhance_existing_areas(output_grid, secondary_color)
    introduce_new_instances(output_grid, secondary_color)
    balance_distribution(output_grid, secondary_color)
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

def analyze_pattern(grid: ColoredGrid, color: int) -> dict:
    rows, cols = grid.get_dimensions()
    quadrants = {1: [0, rows//2, 0, cols//2],
                 2: [0, rows//2, cols//2, cols],
                 3: [rows//2, rows, 0, cols//2],
                 4: [rows//2, rows, cols//2, cols]}
    
    analysis = {q: sum(1 for r in range(quadrants[q][0], quadrants[q][1])
                         for c in range(quadrants[q][2], quadrants[q][3])
                         if grid.get_cell(r, c) == color)
                for q in quadrants}
    
    return analysis

def enhance_existing_areas(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) in [0, 1]:
                        if forms_l_shape(grid, r, c, nr, nc, color) or forms_small_cluster(grid, r, c, nr, nc, color):
                            grid.set_cell(nr, nc, color)

def introduce_new_instances(grid: ColoredGrid, color: int):
    rows, cols = grid.get_dimensions()
    for r in range(0, rows, 3):
        for c in range(0, cols, 3):
            if not any(grid.get_cell(r+dr, c+dc) == color for dr in range(3) for dc in range(3)):
                if is_valid_cell(grid, r, c) and is_valid_cell(grid, r+1, c+1):
                    grid.set_cell(r, c, color)
                    grid.set_cell(r+1, c if r % 2 == 0 else c+1, color)

def balance_distribution(grid: ColoredGrid, color: int):
    analysis = analyze_pattern(grid, color)
    target = sum(analysis.values()) // 4
    
    rows, cols = grid.get_dimensions()
    quadrants = {1: [0, rows//2, 0, cols//2],
                 2: [0, rows//2, cols//2, cols],
                 3: [rows//2, rows, 0, cols//2],
                 4: [rows//2, rows, cols//2, cols]}
    
    for q, count in analysis.items():
        while count < target:
            r_start, r_end, c_start, c_end = quadrants[q]
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if grid.get_cell(r, c) in [0, 1] and not has_adjacent_color(grid, r, c, color):
                        grid.set_cell(r, c, color)
                        count += 1
                        if count == target:
                            break
                if count == target:
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

def forms_small_cluster(grid: ColoredGrid, r1: int, c1: int, r2: int, c2: int, color: int) -> bool:
    count = sum(1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                if is_valid_cell(grid, r1+dr, c1+dc) and grid.get_cell(r1+dr, c1+dc) == color)
    return count >= 2

def has_adjacent_color(grid: ColoredGrid, r: int, c: int, color: int) -> bool:
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if is_valid_cell(grid, nr, nc) and grid.get_cell(nr, nc) == color:
            return True
    return False
