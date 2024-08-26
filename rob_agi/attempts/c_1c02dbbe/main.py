from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_1c02dbbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored regions based on seed points.
    
    The function identifies seed points (non-gray, non-black cells), sorts them
    from left to right and top to bottom, then expands each color horizontally
    and vertically. The expansion respects the order of seed points and allows
    for L-shaped expansions when the same color appears in different columns.
    
    Steps:
    1. Find and sort seed points
    2. Process seed points from left to right
    3. Expand colors horizontally and vertically
    4. Handle same-color seeds in different columns
    5. Preserve the black border
    6. Final cleanup of isolated gray cells
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with expanded color regions
    """
    output_grid = input_grid.deep_copy()
    seed_points = find_and_sort_seed_points(output_grid)
    process_seed_points(output_grid, seed_points)
    preserve_border(output_grid)
    cleanup_isolated_gray(output_grid)
    return output_grid

def find_and_sort_seed_points(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    seed_points = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color not in [0, 5]:  # Not black or gray
                seed_points.append((color, r, c))
    return sorted(seed_points, key=lambda x: (x[2], x[1]))  # Sort by column, then row

def process_seed_points(grid: ColoredGrid, seed_points: List[Tuple[int, int, int]]):
    rows, cols = grid.get_dimensions()
    last_col = -1
    for i, (color, row, col) in enumerate(seed_points):
        if col != last_col:
            right_boundary = cols - 1 if i == len(seed_points) - 1 else (col + seed_points[i+1][2]) // 2
            expand_vertically(grid, color, col)
            expand_horizontally(grid, color, col, right_boundary)
            last_col = col

def expand_vertically(grid: ColoredGrid, color: int, col: int):
    rows, _ = grid.get_dimensions()
    for r in range(1, rows - 1):  # Exclude border rows
        if grid.get_cell(r, col) in [0, 5]:  # Only replace black or gray cells
            grid.set_cell(r, col, color)

def expand_horizontally(grid: ColoredGrid, color: int, start_col: int, end_col: int):
    rows, _ = grid.get_dimensions()
    for r in range(1, rows - 1):  # Exclude border rows
        for c in range(start_col, end_col + 1):
            if grid.get_cell(r, c) in [0, 5]:  # Only replace black or gray cells
                grid.set_cell(r, c, color)

def preserve_border(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        grid.set_cell(r, 0, 0)
        grid.set_cell(r, cols - 1, 0)
    for c in range(cols):
        grid.set_cell(0, c, 0)
        grid.set_cell(rows - 1, c, 0)

def cleanup_isolated_gray(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid.get_cell(r, c) == 5:  # Gray cell
                surrounding_colors = set([
                    grid.get_cell(r-1, c), grid.get_cell(r+1, c),
                    grid.get_cell(r, c-1), grid.get_cell(r, c+1)
                ]) - {0, 5}  # Exclude black and gray
                if len(surrounding_colors) == 1:
                    grid.set_cell(r, c, surrounding_colors.pop())
