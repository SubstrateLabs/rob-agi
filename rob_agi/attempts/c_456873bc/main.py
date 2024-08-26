from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_456873bc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Removes all green (3) areas, replacing them with black (0).
    2. Identifies existing red (2) patterns.
    3. Creates mirrored versions of red patterns across vertical and horizontal axes.
    4. Converts endpoints and significant intersections of red patterns to blue (8).
    5. Ensures overall symmetry in the final grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying all rules.
    """
    grid = input_grid.deep_copy()
    grid = remove_green_areas(grid)
    red_patterns = identify_red_patterns(grid)
    grid = create_mirrored_patterns(grid, red_patterns)
    grid = mark_endpoints_and_intersections(grid)
    grid = ensure_symmetry(grid)
    return grid

def remove_green_areas(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3:
                grid.set_cell(r, c, 0)
    return grid

def identify_red_patterns(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    rows, cols = grid.get_dimensions()
    patterns = []
    visited = set()

    def dfs(r: int, c: int) -> Set[Tuple[int, int]]:
        pattern = set()
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == 2:
                visited.add((curr_r, curr_c))
                pattern.add((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return pattern

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) == 2:
                pattern = dfs(r, c)
                if len(pattern) > 1:
                    patterns.append(pattern)

    return patterns

def create_mirrored_patterns(grid: ColoredGrid, patterns: List[Set[Tuple[int, int]]]) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2

    for pattern in patterns:
        # Mirror vertically
        for r, c in pattern:
            mirrored_c = cols - 1 - c
            if grid.get_cell(r, mirrored_c) == 0:
                grid.set_cell(r, mirrored_c, 2)

        # Mirror horizontally
        for r, c in pattern:
            mirrored_r = rows - 1 - r
            if grid.get_cell(mirrored_r, c) == 0:
                grid.set_cell(mirrored_r, c, 2)

        # Mirror diagonally
        for r, c in pattern:
            mirrored_r, mirrored_c = rows - 1 - r, cols - 1 - c
            if grid.get_cell(mirrored_r, mirrored_c) == 0:
                grid.set_cell(mirrored_r, mirrored_c, 2)

    return grid

def mark_endpoints_and_intersections(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:
                red_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                    if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.get_cell(r + dr, c + dc) == 2)
                if red_neighbors == 1 or (red_neighbors == 2 and (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)):
                    grid.set_cell(r, c, 8)
    return grid

def ensure_symmetry(grid: ColoredGrid) -> ColoredGrid:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            value = grid.get_cell(r, c)
            mirrored_r, mirrored_c = rows - 1 - r, cols - 1 - c
            mirrored_value = grid.get_cell(mirrored_r, mirrored_c)
            if value != mirrored_value:
                if value in (2, 8) and mirrored_value == 0:
                    grid.set_cell(mirrored_r, mirrored_c, value)
                elif mirrored_value in (2, 8) and value == 0:
                    grid.set_cell(r, c, mirrored_value)
    return grid
